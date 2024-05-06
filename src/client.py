# ----- ----- ----- ----- -----
# TODO: FIX SRC IMPORTS IN A BETTER WAY
import sys
from pathlib import Path

import flwr as fl
import torch

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
                                   aux_seed=args.current_client, generate_aux_set=args.scheduler_name == "neve")
    train_loaders, val_loaders, test_loader, aux_loader = prepare_data(train, test, aux, num_clients=args.num_clients,
                                                                       seed=args.seed, batch_size=args.batch_size)
    # Memory optimization
    train_loader = train_loaders[args.current_client]
    val_loader = val_loaders[args.current_client]

    if args.scheduler_name == "neq":
        aux_loader = val_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "cuda" else "cpu"
    # Define Client
    client = get_client(train_loader, val_loader, test_loader, aux_loader, dataset_name=args.dataset_name,
                        use_groupnorm=args.model_use_groupnorm, groupnorm_channels=args.model_groupnorm_groups,
                        model_name=args.model_name, device=device,
                        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, amp=args.amp,
                        neve_use_lr_scheduler=args.neve_use_lr_scheduler,
                        neve_momentum=args.neve_momentum, neve_epsilon=args.neve_epsilon,
                        neve_alpha=args.neve_alpha, neve_delta=args.neve_delta,
                        neve_only_last_layer=args.neve_only_ll,
                        client_id=args.current_client, scheduler_name=args.scheduler_name, use_disk=False)
    fl.client.start_client(server_address=args.server_address, client=client.to_client())


if __name__ == "__main__":
    main(get_args("client"))
