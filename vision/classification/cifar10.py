"""Train CIFAR10 with PyTorch."""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# check if GPU is available,otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define data transformation
print("==> Preparing data..")
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

# load CIFAR10 datasets
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

# classes in the CIFAR10 dataset
classes = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

# building model
print("==> Building model..")
model = models.resnet18(num_classes=10, weights=None)
model.to(device)

# if GPU is available, use DataParallel to increase the training speed
if torch.cuda.is_available():
    model = nn.DataParallel(model)
    cudnn.benchmark = True

# define loss function, optimizer and learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)


def train():
    # put the model in training mode
    model.train()

    # initialize variables
    step = 0
    running_loss = 0.0

    for images, true_labels in train_loader:
        # transfer data to the device the model is using
        images, true_labels = images.to(device), true_labels.to(device)

        optimizer.zero_grad()  # zero the parameter gradients
        outputs = model(images)  # make predictions using the model
        loss = criterion(outputs, true_labels)  # calculate the loss

        # backpropagate the loss to update the model parameters
        loss.backward()
        optimizer.step()

        # update variables
        step += 1
        running_loss += loss.item()

        # print the average training loss every 100 batches
        if step % 100 == 0:
            running_loss = running_loss / 100
            print(f"[{step}/{len(train_loader)}] training loss: {running_loss:.3f}")
            running_loss = 0.0


@torch.no_grad()
def evaluate():
    """
    Evaluate the model's accuracy on the test set.
    """
    model.eval()  # put the model in evaluation mode
    correct = 0
    total = 0

    for images, true_labels in test_loader:
        images, true_labels = images.to(device), true_labels.to(device)

        outputs = model(images)
        _, pred_labels = torch.max(outputs.data, 1)

        total += true_labels.size(0)
        correct += (pred_labels == true_labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    for epoch in range(100):
        print(f"\n[Epoch: {epoch} / 100] [LR: {lr_scheduler.get_last_lr()[0]:.5f}]")
        train()
        evaluate()
        lr_scheduler.step()
