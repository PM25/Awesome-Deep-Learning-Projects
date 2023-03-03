"""WNUT 2017 named entity recognition (token classification) with PyTorch."""
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset

# check if GPU is available,otherwise use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Epoch = 1
BATCH_SIZE = 4
LR = 2e-5
LR_WARMUP_STEPS = 200
WEIGHT_DECAY = 5e-4
LOGGING_STEPS = 100

# load WNUT 2017 dataset
print("\n==> Preparing data..")
wnut17_dataset = load_dataset("wnut_17")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# tokenize and sets the label for each token in the dataset
def preprocess_fn(examples):
    tokenized_examples = tokenizer(
        examples["tokens"], truncation=True, padding="max_length", is_split_into_words=True
    )

    labels = []
    for idx in range(len(examples["tokens"])):
        word_idxes = tokenized_examples.word_ids(
            batch_index=idx
        )  # map each token in the example to its corresponding word
        tags = examples["ner_tags"][idx]  # get the NER tags for the current example

        # create a list to store the labels for each token in the current example
        label = []
        previous_word_idx = None
        for word_idx in word_idxes:
            # if the current token is the first token in a given word, label it with the corresponding NER tag
            if word_idx is not None and word_idx != previous_word_idx:
                label.append(tags[word_idx])
            else:  # otherwise, set the label to -100
                label.append(-100)
            previous_word_idx = word_idx
        labels.append(label)

    tokenized_examples["labels"] = labels

    return tokenized_examples


tokenized_wnut17_dataset = wnut17_dataset.map(preprocess_fn, batched=True)
tokenized_wnut17_dataset = tokenized_wnut17_dataset.with_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)
train_dataset = tokenized_wnut17_dataset["train"]
test_dataset = tokenized_wnut17_dataset["test"]

# create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

label_names = wnut17_dataset["train"].features[f"ner_tags"].feature.names
id2label = {idx: label for idx, label in enumerate(label_names)}
# id2label = {0: 'O', 1: 'B-corporation', 2: 'I-corporation', 3: 'B-creative-work', ..., 11: 'B-product', 12: 'I-product'}

# build the model
print("\n==> Building model..")
model = AutoModelForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=13)

# if GPU is available, use DataParallel to increase the training speed
if torch.cuda.is_available():
    model = nn.DataParallel(model)
    cudnn.benchmark = True

# define the optimizer and learning rate scheduler
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
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
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        true_labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()  # zero the parameter gradients
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=true_labels
        )  # make predictions
        loss = outputs["loss"]

        # backpropagate the loss to update the model parameters
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # calculate training accuracy
        logits = outputs["logits"]
        _, pred_labels = torch.max(logits.data, 2)

        label_idxes = true_labels.flatten() != -100
        pred_labels = pred_labels.flatten()[label_idxes]
        true_labels = true_labels.flatten()[label_idxes]

        correct = (pred_labels == true_labels).sum().item()
        train_acc += correct / true_labels.size(0)

        # update step and training loss
        step += 1
        train_loss += loss.item()

        # print the average loss every LOGGING_STEPS steps
        if step % LOGGING_STEPS == 0:
            train_loss /= LOGGING_STEPS
            train_acc /= LOGGING_STEPS
            print(
                f"[{step}/{len(train_loader)}][LR: {lr_scheduler.get_last_lr()[0]:.7f}] training loss = {train_loss:.3f}, training acc = {train_acc:.3f}"
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
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        true_labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=true_labels
        )  # make predictions
        logits = outputs["logits"]
        _, pred_labels = torch.max(logits.data, 2)

        label_idxes = true_labels.flatten() != -100
        pred_labels = pred_labels.flatten()[label_idxes]
        true_labels = true_labels.flatten()[label_idxes]

        total += true_labels.size(0)
        correct += (pred_labels == true_labels).sum().item()

    print(f"Validation Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    for epoch in range(Epoch):
        print(f"\n[Epoch: {epoch} / {Epoch}] [LR: {lr_scheduler.get_last_lr()[0]:.7f}]")
        train()
        evaluate()
