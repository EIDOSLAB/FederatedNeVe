import flwr as fl
import torch

from arguments import get_args
from dataloaders import get_dataset, split_data
from my_flwr.clients import CifarClient
from my_flwr.strategies import weighted_average_fit, weighted_average_eval
from models.test import Net

train_loaders, val_loaders, test_loader = None, None, None


def client_fn(cid: str):
    assert train_loaders and val_loaders and test_loader
    print("Created client with cid:", cid)
    # Load data from the client
    train_loader = train_loaders[int(cid) % len(train_loaders)]
    # TODO: WHEN AUX_LOADER IS IMPLEMENTED PASS THE AUX_LOADER INSTEAD OF THE VALIDATION ONE
    valid_loader = val_loaders[int(cid) % len(val_loaders)]
    return CifarClient(train_loader, test_loader, aux_loader=valid_loader)


def main(args):
    # TODO: this is a really bad way to do this, for now it is acceptable
    global train_loaders, val_loaders, test_loader
    # Initialize global model and data
    train, test = get_dataset(args.dataset_root, args.dataset_name)
    train_loaders, val_loaders, test_loader = split_data(train, test, num_clients=args.num_clients)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Performing training on:", device)
    net = Net().to(device)

    client_resources = None
    if "cuda" in device.type:
        client_resources = {"num_cpus": 1, "num_gpus": 1 / args.num_clients}

    # Launch the simulation
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,  # A function to run a _virtual_ client when required
        num_clients=args.num_clients,  # Total number of clients available
        config=fl.server.ServerConfig(num_rounds=args.epochs),  # Specify number of FL rounds
        strategy=fl.server.strategy.FedAvg(fit_metrics_aggregation_fn=weighted_average_fit,
                                           evaluate_metrics_aggregation_fn=weighted_average_eval),  # A Flower strategy
        client_resources=client_resources
    )


if __name__ == "__main__":
    main(get_args("simulation"))
