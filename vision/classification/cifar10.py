import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Check if GPU is available, otherwise use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # define the data transformation
    data_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # load CIFAR10 training and testing datasets
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=data_transform)
    train_loader = utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=data_transform)
    test_loader = utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # classes in the CIFAR10 dataset
    classes = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

    # define model, loss function and optimizer
    model = models.resnet18(num_classes=10, weights=None).to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)

    print("\nStart Training")

    for epoch in range(10):  # loop over the dataset multiple times
        step = 0
        running_loss = 0.0

        for images, labels in train_loader:
            # transfer data to the device the model is using
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()  # zero the parameter gradients
            outputs = model(images)  # make predictions using the model
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print training loss every 100 batches
            step += 1
            running_loss += loss.item()
            if step % 100 == 0:
                running_loss = running_loss / 100
                print(f"[epoch:{epoch + 1}][{step}/{len(train_loader)}] training loss: {running_loss:.3f}")
                running_loss = 0.0

        evaluate(model, test_loader)

    print("Finished Training")

    evaluate(model, test_loader)
    plot_images_and_preds(model, test_loader, classes)


@torch.no_grad()
def evaluate(model, test_loader):
    """
    Evaluate the model's accuracy on the test set.
    """
    model = model.eval()  # put the model in evaluation mode
    correct = 0
    total = 0

    for images, true_labels in test_loader:
        images, true_labels = images.to(DEVICE), true_labels.to(DEVICE)

        outputs = model(images)
        _, pred_labels = torch.max(outputs.data, 1)

        total += true_labels.size(0)
        correct += (pred_labels == true_labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")


def plot_images_and_preds(model, test_loader, classes):
    """
    Plot 9 randomly selected images from the test dataset along with their true labels and predicted labels.
    Green title means the prediction was correct, red title means it was incorrect.
    """
    fig, axs = plt.subplots(3, 3, figsize=(12, 12))

    # get a batch of test images and their labels
    images, true_labels = next(iter(test_loader))

    outputs = model(images.to(DEVICE))
    _, pred_labels = torch.max(outputs, 1)

    # plot each image and its prediction
    for image, true_label, pred_label, ax in zip(images, true_labels, pred_labels, axs.flatten()):
        image = image / 2 + 0.5  # unnormalize the image
        image_np = image.cpu().numpy()

        ax.imshow(np.transpose(image_np, (1, 2, 0)))  # Plot the image
        title_color = "green" if true_label == pred_label else "red"
        ax.set_title(f"True: {classes[true_label]} | Predicted: {classes[pred_label]}", color=title_color)

    plt.savefig("./predictions.png")
    print("Plot saved to ./predictions.png")


if __name__ == "__main__":
    main()
