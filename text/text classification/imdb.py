"""Train IMDB text classification with PyTorch."""
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

# check if GPU is available,otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Epoch = 2
BATCH_SIZE = 4
LR = 2e-5
LR_WARMUP_STEPS = 1000
LOGGING_STEPS = 100

# Load IMDB dataset
print("\n==> Preparing data..")
imdb_dataset = load_dataset("imdb")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize the dataset
def tokenize_fn(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length")


tokenized_imdb_dataset = imdb_dataset.map(tokenize_fn, batched=True)
tokenized_imdb_dataset = tokenized_imdb_dataset.with_format(
    type="torch", columns=["input_ids", "attention_mask", "label"]
)
train_dataset = tokenized_imdb_dataset["train"]
test_dataset = tokenized_imdb_dataset["test"]

# create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

id2label = {0: "negative", 1: "positive"}

# build the model
print("\n==> Building model..")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# if GPU is available, use DataParallel to increase the training speed
if torch.cuda.is_available():
    model = nn.DataParallel(model)
    cudnn.benchmark = True

# define the optimizer and learning rate scheduler
optimizer = optim.AdamW(model.parameters(), lr=LR)
lr_scheduler = transformers.get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=LR_WARMUP_STEPS, num_training_steps=Epoch * len(train_loader)
)


def train():
    """
    Train the model.
    """
    model.train()  # put the model in training mode

    step = 0
    train_loss = 0.0
    train_acc = 0.0

    for batch in train_loader:
        # transfer data to the device the model is using
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        true_labels = batch["label"].to(device)

        optimizer.zero_grad()  # zero the parameter gradients
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=true_labels
        )  # make predictions
        loss = outputs["loss"]

        # backpropagate the loss to update the model parameters
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # update step and training loss
        step += 1
        train_loss += loss.item()

        # calculate training accuracy
        logits = outputs["logits"]
        _, pred_labels = torch.max(logits.data, 1)
        correct = (pred_labels == true_labels).sum().item()
        train_acc += correct / true_labels.size(0)

        # print the average loss every LOGGING_STEPS steps
        if step % LOGGING_STEPS == 0:
            train_loss /= LOGGING_STEPS
            train_acc /= LOGGING_STEPS
            print(
                f"[{step}/{len(train_loader)}][LR: {lr_scheduler.get_last_lr()[0]:.7f}] training loss = {train_loss:.3f}, training acc = {train_acc:.2f}"
            )
            train_loss = train_acc = 0.0


@torch.no_grad()
def evaluate():
    """
    Evaluate the model's accuracy on the test set.
    """
    model.eval()  # put the model in evaluation mode
    correct = 0
    total = 0

    for batch in tqdm(test_loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        true_labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=true_labels
        )  # make predictions
        logits = outputs["logits"]
        _, pred_labels = torch.max(logits.data, 1)

        total += true_labels.size(0)
        correct += (pred_labels == true_labels).sum().item()

    print(f"Validation Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    for epoch in range(Epoch):
        print(f"\n[Epoch: {epoch} / {Epoch}] [LR: {lr_scheduler.get_last_lr()[0]:.7f}]")
        train()
        evaluate()
