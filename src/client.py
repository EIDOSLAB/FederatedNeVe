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
from src.my_flwr.clients import get_client
from src.utils import set_seeds


def main(args):
    # Init seeds
    set_seeds(args.seed)
    # Load data
    train, test, aux = get_dataset(args.dataset_root, args.dataset_name,
                                   aux_seed=args.current_client, generate_aux_set=args.use_neve)
    train_loaders, val_loaders, test_loader, aux_loader = prepare_data(train, test, aux, num_clients=args.num_clients,
                                                                       seed=args.seed, batch_size=args.batch_size)
    # Memory optimization
    train_loader = train_loaders[args.current_client]
    val_loader = val_loaders[args.current_client]
    del train_loaders, val_loaders, train, test, aux
    # Define Client
    client = get_client(train_loader, val_loader, test_loader, aux_loader, dataset_name=args.dataset_name,
                        use_groupnorm=args.model_use_groupnorm,
                        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, amp=args.amp,
                        neve_momentum=args.neve_momentum, neve_epsilon=args.neve_epsilon,
                        neve_alpha=args.neve_alpha, neve_delta=args.neve_delta,
                        client_id=args.current_client, use_neve=args.use_neve, neve_use_disk=False)
    fl.client.start_client(server_address=args.server_address, client=client.to_client())


if __name__ == "__main__":
    main(get_args("client"))
