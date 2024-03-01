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
from src.my_flwr.clients import CifarDefaultClient
from src.utils import set_seeds


def main(args):
    # Init seeds
    set_seeds(args.seed)
    # Load data
    train, test, aux = get_dataset(args.dataset_root, args.dataset_name, seed=args.seed, generate_aux_set=args.use_neve)
    train_loaders, val_loaders, test_loader, aux_loader = prepare_data(train, test, aux, num_clients=args.num_clients,
                                                                       seed=args.seed, batch_size=args.batch_size)

    # TODO: generate a get_client() function that returns the correct Client when we will try neve
    fl.client.start_client(server_address=args.server_address,
                           client=CifarDefaultClient(train_loader=train_loaders[args.current_client],
                                                     valid_loader=val_loaders[args.current_client],
                                                     test_loader=test_loader,
                                                     model_name=args.model_name,
                                                     dataset_name=args.dataset_name,
                                                     optimizer_name=args.optimizer,
                                                     lr=args.lr,
                                                     momentum=args.momentum,
                                                     weight_decay=args.weight_decay,
                                                     amp=args.amp,
                                                     client_id=args.current_client).to_client()
                           )


if __name__ == "__main__":
    main(get_args("client"))
