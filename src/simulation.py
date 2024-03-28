# ----- ----- ----- ----- -----
# TODO: FIX SRC IMPORTS IN A BETTER WAY
import os
import shutil
import sys
from pathlib import Path

import flwr as fl
import torch
import wandb

FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# ----- ----- ----- ----- -----

from src.arguments import get_args
from src.dataloaders import get_dataset, prepare_data, load_aux_dataset
from src.my_flwr.clients import get_client
from src.my_flwr.strategies import weighted_average_fit, weighted_average_eval
from src.utils import set_seeds
from src.NeVe.federated.flwr.strategies import FedNeVeAvg

dataset_name, use_groupnorm = "", True
train_loaders, val_loaders, test_loader, aux_loaders = None, None, None, None

use_neve, neve_use_disk = True, True
neve_disk_folder = os.path.join("../neve_data/")
neve_momentum, neve_epsilon, neve_alpha, neve_delta = 0.5, 1e-3, 0.5, 10


def client_fn(cid: str):
    assert train_loaders and val_loaders and test_loader and aux_loaders
    print("Created client with cid:", cid)
    # Load data from the client
    train_loader = train_loaders[int(cid) % len(train_loaders)]
    valid_loader = val_loaders[int(cid) % len(val_loaders)]
    aux_loader = aux_loaders[int(cid) % len(aux_loaders)]
    # TODO: add these params
    """
    model_name = args.model_name,
    optimizer_name = args.optimizer,
    lr = args.lr,
    momentum = args.momentum,
    weight_decay = args.weight_decay,
    amp = args.amp,
    """
    return get_client(train_loader, valid_loader, test_loader, aux_loader, dataset_name,
                      use_groupnorm=use_groupnorm, client_id=int(cid),
                      neve_momentum=neve_momentum, neve_epsilon=neve_epsilon,
                      neve_alpha=neve_alpha, neve_delta=neve_delta,
                      use_neve=use_neve, neve_use_disk=neve_use_disk, neve_disk_folder=neve_disk_folder).to_client()


def main(args):
    # TODO: this is a really bad way to do this, for now it is acceptable
    global dataset_name, use_groupnorm, train_loaders, val_loaders, test_loader, aux_loaders
    global neve_epsilon, neve_momentum, neve_alpha, neve_delta
    global use_neve, neve_use_disk
    neve_epsilon = args.neve_epsilon
    neve_momentum = args.neve_momentum
    neve_alpha = args.neve_alpha
    neve_delta = args.neve_delta
    dataset_name = args.dataset_name.lower()
    use_groupnorm = args.model_use_groupnorm
    use_neve = args.scheduler_name == "neve"
    neve_use_disk = True
    # Cleanup neve_disk_folder
    if os.path.exists(neve_disk_folder):
        shutil.rmtree(neve_disk_folder)
    # Init seeds
    set_seeds(args.seed)
    # Initialize global model and data
    train, test, _ = get_dataset(args.dataset_root, args.dataset_name,
                                 aux_seed=args.seed, generate_aux_set=False)
    train_loaders, val_loaders, test_loader, _ = prepare_data(train, test, None, num_clients=args.num_clients,
                                                              seed=args.seed, batch_size=args.batch_size)
    # Generate aux loaders
    aux_loaders = [load_aux_dataset(shape=(3, 24, 24), aux_seed=idx, batch_size=args.batch_size)
                   for idx in range(args.num_clients)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Performing training on:", device)

    client_resources = None
    if "cuda" in device.type:
        client_resources = {"num_cpus": 1, "num_gpus": 1 / args.num_clients}

    # Init wandb project
    # TODO REMOVE COMMENT
    wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, config=args)

    # Select strategy
    strategy = FedNeVeAvg if args.scheduler_name == "neve" else fl.server.strategy.FedAvg

    # Launch the simulation
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,  # A function to run a _virtual_ client when required
        num_clients=args.num_clients,  # Total number of clients available
        config=fl.server.ServerConfig(num_rounds=args.epochs),  # Specify number of FL rounds
        strategy=strategy(fit_metrics_aggregation_fn=weighted_average_fit,
                          min_fit_clients=args.num_clients,
                          min_evaluate_clients=args.num_clients,
                          min_available_clients=args.num_clients,
                          evaluate_metrics_aggregation_fn=weighted_average_eval),
        client_resources=client_resources,
    )
    # Save model...

    # End wandb run
    wandb.run.finish()


if __name__ == "__main__":
    main(get_args("simulation"))
