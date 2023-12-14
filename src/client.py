import flwr as fl

from dataloaders import get_cifar_10, split_data
from my_flwr.clients import CifarClient


def main(num_clients: int = 1, current_client: int = 0):
    # Load data (es cifar10,100)
    train, test = get_cifar_10("../datasets/")  # get_cifar_100("..datasets/")
    train_loaders, val_loaders, test_loader = split_data(train, test, num_clients=num_clients)
    # TODO: WHEN RANDOM_DATA GENERATION IS IMPLEMENTED PASS THE AUX_LOADER INSTEAD OF THE VALIDATION ONE
    fl.client.start_numpy_client(server_address="127.0.0.1:8080",
                                 client=CifarClient(train_loaders[current_client], test_loader,
                                                    aux_loader=val_loaders[current_client]))


if __name__ == "__main__":
    # Load model and data
    clients = 2  # Number of splits of the data (1 per client, or just to test with a subset of data)
    current = 1  # ID of the client for the current simulation (should be changed for each one, < num_clients)
    assert 0 <= current < clients
    main(clients, current)
