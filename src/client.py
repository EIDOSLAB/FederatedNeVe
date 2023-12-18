import flwr as fl

from arguments import get_args
from dataloaders import get_dataset, split_data
from my_flwr.clients import CifarClient


def main(args):
    # Load data
    train, test = get_dataset(args.dataset_root, args.dataset_name)
    train_loaders, val_loaders, test_loader = split_data(train, test, num_clients=args.num_clients)
    # TODO: WHEN RANDOM_DATA GENERATION IS IMPLEMENTED PASS THE AUX_LOADER INSTEAD OF THE VALIDATION ONE
    fl.client.start_numpy_client(server_address=args.server_address,
                                 client=CifarClient(train_loaders[args.current_client], test_loader,
                                                    aux_loader=val_loaders[args.current_client],
                                                    client_id=args.current_client))


if __name__ == "__main__":
    main(get_args("client"))
