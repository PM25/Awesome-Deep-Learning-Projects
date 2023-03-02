"""CIFAR10 image classification with PyTorch."""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# check if GPU is available,otherwise use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define hyperparameters
EPOCHS = 30
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 128
LOGGING_STEPS = 100

# define data transformation
print("\n==> Preparing data..")
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# load CIFAR10 datasets
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# classes in the CIFAR10 dataset
id2label = {0: "Plane", 1: "Car", 2: "Bird", 3: "Cat", 4: "Deer", 5: "Dog", 6: "Frog", 7: "Horse", 8: "Ship", 9: "Truck"}

# build the model
print("\n==> Building model..")
model = models.resnet18(num_classes=10, weights=None)
model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
model.maxpool = nn.Identity()
model.to(DEVICE)

# if GPU is available, use DataParallel to increase the training speed
if torch.cuda.is_available():
    model = nn.DataParallel(model)
    cudnn.benchmark = True

# define the loss function, optimizer and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)


def train():
    """
    Train the model.
    """
    model.train()  # put the model in training mode

    step = 0
    train_loss = 0.0

    for images, true_labels in train_loader:
        # transfer data to the device the model is using
        images, true_labels = images.to(DEVICE), true_labels.to(DEVICE)

        optimizer.zero_grad()  # zero the parameter gradients
        outputs = model(images)  # make predictions
        loss = criterion(outputs, true_labels)  # calculate loss

        # backpropagate the loss to update the model parameters
        loss.backward()
        optimizer.step()

        # update step and training loss
        step += 1
        train_loss += loss.item()

        # print the average loss every LOGGING_STEPS steps
        if step % LOGGING_STEPS == 0:
            train_loss = train_loss / LOGGING_STEPS
            print(f"[{step}/{len(train_loader)}] training loss = {train_loss:.3f}")
            train_loss = 0.0


@torch.no_grad()
def evaluate():
    """
    Evaluate the model's accuracy on the test set.
    """
    model.eval()  # put the model in evaluation mode
    correct = 0
    total = 0

    for images, true_labels in test_loader:
        images, true_labels = images.to(DEVICE), true_labels.to(DEVICE)

        outputs = model(images)
        _, pred_labels = torch.max(outputs.data, 1)

        total += true_labels.size(0)
        correct += (pred_labels == true_labels).sum().item()

    print(f"Validation Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    for epoch in range(EPOCHS):
        print(f"\n[Epoch: {epoch} / {EPOCHS}] [LR: {lr_scheduler.get_last_lr()[0]:.5f}]")
        train()
        evaluate()
        lr_scheduler.step()
