# ----- ----- ----- ----- -----
# TODO: FIX SRC IMPORTS IN A BETTER WAY
import sys
from pathlib import Path

import flwr as fl

FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# ----- ----- ----- ----- -----

from src.arguments import get_args
from src.dataloaders import get_dataset, prepare_data
from src.my_flwr.clients import CifarCustomClient
from src.utils import set_seeds


def main(args):
    # Init seeds
    set_seeds(args.seed)
    # Load data
    train, test = get_dataset(args.dataset_root, args.dataset_name)
    train_loaders, val_loaders, test_loader = prepare_data(train, test, seed=args.seed, num_clients=args.num_clients,
                                                           batch_size=args.batch_size)

    # TODO: WHEN RANDOM_DATA GENERATION IS IMPLEMENTED PASS THE AUX_LOADER INSTEAD OF THE VALIDATION ONE
    fl.client.start_client(server_address=args.server_address,
                           client=CifarCustomClient(train_loader=train_loaders[args.current_client],
                                                    valid_loader=val_loaders[args.current_client],
                                                    test_loader=test_loader,
                                                    aux_loader=val_loaders[args.current_client],
                                                    client_id=args.current_client,
                                                    neve_epsilon=args.neve_epsilon,
                                                    neve_momentum=args.neve_momentum).to_client()
                           )


if __name__ == "__main__":
    main(get_args("client"))
