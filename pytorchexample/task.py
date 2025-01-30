"""pytorchexample: A Flower / PyTorch app."""

from collections import OrderedDict
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Net(nn.Module):
    """Simple CNN for classification."""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # 32 filters
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def get_weights(net):
    """Extract PyTorch model weights as a list of NumPy arrays."""
    weights = [param.data.cpu().numpy() for param in net.parameters()]
    print(f"Extracted {len(weights)} layers of weights.")
    return weights


def set_weights(net, parameters):
    """Set PyTorch model weights from a list of NumPy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

    # Validate the keys
    expected_keys = list(net.state_dict().keys())
    received_keys = list(state_dict.keys())

    if expected_keys != received_keys:
        raise ValueError(
            f"Mismatch in keys:\n"
            f"Expected keys: {expected_keys}\n"
            f"Received keys: {received_keys}"
        )

    net.load_state_dict(state_dict, strict=True)


def load_data(partition_id: int, num_partitions: int, batch_size: int, rotation_degree: int):
    """Load pre-saved rotated MNIST data."""
    data_dir = "./pytorchexample/data/rotated_mnist"

    # Verify that the data files exist
    train_path = os.path.join(data_dir, f"client_{partition_id}_train.pt")
    test_path = os.path.join(data_dir, f"client_{partition_id}_test.pt")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Data files for client {partition_id} not found in {data_dir}")

    # Load pre-saved partitions
    train_data = torch.load(train_path)
    test_data = torch.load(test_path)

    # Create DataLoaders
    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_data, batch_size=batch_size)
    return trainloader, testloader


def train(net, trainloader, valloader, epochs, lr, device):
    """Train the network."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.to(device)
    net.train()

    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.3f}")

    print("Finished Training")
    return {"loss": running_loss / len(trainloader)}


def test(net, testloader, device):
    """Evaluate the network on the test set."""
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0
    net.to(device)
    net.eval()

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            loss += criterion(outputs, labels).item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Test Loss: {loss / len(testloader):.3f}, Accuracy: {accuracy:.3f}")
    return loss / len(testloader), accuracy
