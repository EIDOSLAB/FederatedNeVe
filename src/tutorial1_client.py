from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

import flwr as fl

from NeVe import NeVeOptimizer


def load_data(num_clients: int = 5, val_percentage: int = 10):
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("../datasets/", train=True, download=True, transform=transform)
    testset = CIFAR10("../datasets/", train=False, download=True, transform=transform)
    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // val_percentage  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=32, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=32))
    testloader = DataLoader(testset, batch_size=32)
    # TODO: IMPORT GENERATION OF RANDOM DATA
    aux_loader = None
    return trainloaders, valloaders, testloader, aux_loader


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CifarClient(fl.client.NumPyClient):
    def __init__(self, train_loader: DataLoader, test_loader: DataLoader, aux_loader: DataLoader):
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Net().to(self.DEVICE)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.aux_loader = aux_loader
        self.neve = NeVeOptimizer(self.model, velocity_momentum=0.5)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # Get the velocity value before the training step (velocity at time t-1)
        self.neve.set_active(True)
        _ = self.test(self.aux_loader)
        self.neve.set_active(False)
        # TODO: init_step=True should be done only the really first time we evaluate the velocity
        _ = self.neve.step(init_step=True)
        # TODO: SAVE NEVE CURRENT_ACTIVATIONS INTO FILE
        loss, accuracy = self.train(self.train_loader, epochs=1)
        return self.get_parameters(config={}), len(self.train_loader), {"loss": float(loss), "accuracy": float(accuracy)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # TODO: READ NEVE CURRENT_ACTIVATIONS FROM FILE
        # Get the velocity value after the training step (velocity at time t)
        self.neve.set_active(True)
        self.test(self.aux_loader)
        self.neve.set_active(False)
        velocity_data = self.neve.step()
        print("Velocity data:", velocity_data["neve"])
        # Validate the model on the testset
        loss, accuracy = self.test(self.test_loader)
        return float(loss), len(self.test_loader), {"loss": float(loss), "accuracy": float(accuracy)}

    def train(self, dataloader, epochs):
        """Train the network on the training set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, total_loss = 0, 0, 0.0
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        for _ in range(epochs):
            for images, labels in dataloader:
                images, labels = images.to(self.DEVICE), labels.to(self.DEVICE)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return total_loss, accuracy

    def test(self, dataloader):
        """Validate the network on the entire test set."""
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for data in dataloader:
                images, labels = data[0].to(self.DEVICE), data[1].to(self.DEVICE)
                outputs = self.model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return loss, accuracy


if __name__ == "__main__":
    # Load model and data
    num_clients = 2 # Number o
    current_client = 1
    trainloaders, validloaders, testloader, auxloader = load_data(num_clients)
    # TODO: WHEN RANDOM_DATA GENERATION IS IMPLEMENTED PASS THE AUX_LOADER INSTEAD OF THE VALIDATION ONE
    fl.client.start_numpy_client(server_address="127.0.0.1:8080",
                                 client=CifarClient(trainloaders[current_client], testloader,
                                                    aux_loader=validloaders[current_client]))
