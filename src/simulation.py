# ----- ----- ----- ----- -----
# TODO: FIX SRC IMPORTS IN A BETTER WAY
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
from src.dataloaders import get_dataset, prepare_data
from src.my_flwr.clients import CifarDefaultClient
from src.my_flwr.strategies import weighted_average_fit, weighted_average_eval
from src.utils import set_seeds

dataset_name, train_loaders, val_loaders, test_loader = "", None, None, None
neve_epsilon, neve_momentum = 1e-3, 0.5


def client_fn(cid: str):
    assert train_loaders and val_loaders and test_loader
    print("Created client with cid:", cid)
    # Load data from the client
    train_loader = train_loaders[int(cid) % len(train_loaders)]
    valid_loader = val_loaders[int(cid) % len(val_loaders)]
    # TODO: ADD AUX LOADER
    return CifarDefaultClient(train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
                              dataset_name=dataset_name, client_id=int(cid)).to_client()


def main(args):
    # TODO: this is a really bad way to do this, for now it is acceptable
    global dataset_name, train_loaders, val_loaders, test_loader
    global neve_epsilon, neve_momentum
    # Init seeds
    set_seeds(args.seed)
    neve_epsilon = args.neve_epsilon
    neve_momentum = args.neve_momentum
    dataset_name = args.dataset_name
    # Initialize global model and data
    train, test, aux = get_dataset(args.dataset_root, args.dataset_name, seed=args.seed, generate_aux_set=args.use_neve)
    train_loaders, val_loaders, test_loader, aux_loader = prepare_data(train, test, aux, num_clients=args.num_clients,
                                                                       seed=args.seed, batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Performing training on:", device)

    client_resources = None
    if "cuda" in device.type:
        client_resources = {"num_cpus": 1, "num_gpus": 1 / args.num_clients}

    # Init wandb project
    wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, config=args)

    # Launch the simulation
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,  # A function to run a _virtual_ client when required
        num_clients=args.num_clients,  # Total number of clients available
        config=fl.server.ServerConfig(num_rounds=args.epochs),  # Specify number of FL rounds
        strategy=fl.server.strategy.FedAvg(fit_metrics_aggregation_fn=weighted_average_fit,
                                           evaluate_metrics_aggregation_fn=weighted_average_eval),  # A Flower strategy
        client_resources=client_resources,
    )
    # Save model...

    # End wandb run
    wandb.run.finish()


if __name__ == "__main__":
    main(get_args("simulation"))
