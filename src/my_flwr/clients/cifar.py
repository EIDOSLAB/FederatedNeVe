from collections import OrderedDict

import flwr as fl
import torch
from torch.utils.data import DataLoader

from NeVe import NeVeOptimizer
from models.test import Net


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
        with self.neve:
            _ = self.test(self.aux_loader)
        # TODO: init_step=True should be done only the really first time we evaluate the velocity
        _ = self.neve.step(init_step=True)
        # TODO: SAVE NEVE CURRENT_ACTIVATIONS INTO FILE
        loss, accuracy = self.train(self.train_loader, epochs=1)
        return self.get_parameters(config={}), len(self.train_loader), {"loss": float(loss),
                                                                        "accuracy": float(accuracy)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # TODO: READ NEVE CURRENT_ACTIVATIONS FROM FILE
        # Get the velocity value after the training step (velocity at time t)
        with self.neve:
            self.test(self.aux_loader)
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
            for data in dataloader:
                images, labels = data[0].to(self.DEVICE), data[1].to(self.DEVICE)
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
