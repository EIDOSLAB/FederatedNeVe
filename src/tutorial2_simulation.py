from typing import List, Tuple

import flwr as fl
import torch
from flwr.common import Metrics

from dataloaders import get_cifar_10, split_data
from flwclients.cifar import CifarClient
from models.test import Net


def client_fn(cid: str):
    print("Created client with cid:", cid)
    # Load data from the client
    trainloader = train_loaders[int(cid) % len(train_loaders)]
    # TODO: WHEN AUX_LOADER IS IMPLEMENTED PASS THE AUX_LOADER INSTEAD OF THE VALIDATION ONE
    validloader = val_loaders[int(cid) % len(val_loaders)]
    return CifarClient(trainloader, test_loader, aux_loader=validloader)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    aggregate_data = {"accuracy": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples)}
    return aggregate_data


def weighted_average_fit(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    print("Fit data to aggregate:", metrics)
    aggregate_data = weighted_average(metrics)
    print("Fit aggregation result:", aggregate_data)
    return aggregate_data


def weighted_average_eval(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    print("Eval data to aggregate:", metrics)
    aggregate_data = weighted_average(metrics)
    print("Eval aggregation result:", aggregate_data)
    return aggregate_data


if __name__ == "__main__":
    num_clients = 2
    # Initialize global model and data
    train, test = get_cifar_10("../datasets/")
    train_loaders, val_loaders, test_loader = split_data(train, test, num_clients=num_clients)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(DEVICE)
    net = Net().to(DEVICE)

    client_resources = None
    if "cuda" in DEVICE.type:
        client_resources = {"num_cpus": 1, "num_gpus": 1 / num_clients}

    # Launch the simulation
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,  # A function to run a _virtual_ client when required
        num_clients=num_clients,  # Total number of clients available
        config=fl.server.ServerConfig(num_rounds=5),  # Specify number of FL rounds
        strategy=fl.server.strategy.FedAvg(fit_metrics_aggregation_fn=weighted_average_fit,
                                           evaluate_metrics_aggregation_fn=weighted_average_eval),  # A Flower strategy
        client_resources=client_resources
    )
