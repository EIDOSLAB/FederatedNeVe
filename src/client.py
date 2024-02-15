import flwr as fl

from src.arguments import get_args
from src.dataloaders import get_dataset, prepare_data
from src.my_flwr.clients import NeVeCifarClient
from src.utils import set_seeds


def main(args):
    # Init seeds
    set_seeds(args.seed)
    # Load data
    train, test = get_dataset(args.dataset_root, args.dataset_name)
    train_loaders, val_loaders, test_loader = prepare_data(train, test, num_clients=args.num_clients)
    # TODO: WHEN RANDOM_DATA GENERATION IS IMPLEMENTED PASS THE AUX_LOADER INSTEAD OF THE VALIDATION ONE
    fl.client.start_numpy_client(server_address=args.server_address,
                                 client=NeVeCifarClient(train_loaders[args.current_client], test_loader,
                                                        aux_loader=val_loaders[args.current_client],
                                                        client_id=args.current_client,
                                                        neve_epsilon=args.neve_epsilon,
                                                        neve_momentum=args.neve_momentum))


if __name__ == "__main__":
    main(get_args("client"))
