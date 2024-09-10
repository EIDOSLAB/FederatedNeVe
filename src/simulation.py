# ----- ----- ----- ----- -----
# TODO: FIX SRC IMPORTS IN A BETTER WAY
import os
import shutil
import sys
from pathlib import Path

import flwr as fl
import torch
from flwr.server.strategy import FedAvg

import wandb

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# ----- ----- ----- ----- -----

from src.arguments import get_args
from src.dataloaders import get_dataset, prepare_data, load_aux_dataset
from src.my_flwr.clients import get_client
from src.my_flwr.strategies import weighted_average_fit, weighted_average_eval
from src.utils import set_seeds
from src.NeVe.federated.flwr.strategies import FedNeVeAvg
from src.NeVe.federated.flwr.strategies.sampler import get_client_sampler

dataset_name, use_groupnorm, groupnorm_channels = "", True, 2
train_loaders, val_loaders, test_loader, aux_loaders = None, None, None, None

scheduler_name, use_disk = "baseline", True
disk_folder = os.path.join("../fclients_data/")
neve_momentum, neve_epsilon, neve_alpha, neve_delta = 0.5, 1e-3, 0.5, 10
base_lr = 0.1
optimizer_name = "sgd"
momentum, weight_decay = 0.9, 5e-4
amp = True
device = "cuda"
model_name = "resnet18"
use_pretrain = False
neve_active = False
neve_multiepoch = False
neve_multiepoch_epochs = 2
neve_only_last_layer = True
neve_use_lr_scheduler = True
neve_use_early_stop = False
leaf_input_dim = 10

dataset_root = ""
train = None
test = None
dataset_iid = True
num_clients = 1
lda_concentration = 0.1
seed = 0
batch_size = 8


def client_fn(cid: str):
    assert aux_loaders
    print("Created client with cid:", cid)
    # Load data from the client
    train_loader, val_loader, test_loader, _ = prepare_data(dataset_root, dataset_name,
                                                            train, test, None,
                                                            split_iid=dataset_iid,
                                                            num_clients=num_clients,
                                                            concentration=lda_concentration,
                                                            seed=seed, batch_size=batch_size,
                                                            current_client=int(cid))
    aux_loader = aux_loaders[int(cid) % len(aux_loaders)]
    return get_client(train_loader, val_loader, test_loader, aux_loader, dataset_name=dataset_name,
                      use_groupnorm=use_groupnorm, groupnorm_channels=groupnorm_channels,
                      use_pretrain=use_pretrain,
                      client_id=int(cid), model_name=model_name, device=device,
                      lr=base_lr, optimizer_name=optimizer_name,
                      momentum=momentum, weight_decay=weight_decay, amp=amp,
                      use_neve=neve_active, use_neve_multiepoch=neve_multiepoch,
                      neve_multiepoch_epochs=neve_multiepoch_epochs,
                      neve_use_lr_scheduler=neve_use_lr_scheduler,
                      neve_use_early_stop=neve_use_early_stop,
                      neve_momentum=neve_momentum, neve_epsilon=neve_epsilon,
                      neve_alpha=neve_alpha, neve_delta=neve_delta,
                      scheduler_name=scheduler_name, use_disk=use_disk,
                      neve_only_last_layer=neve_only_last_layer,
                      disk_folder=disk_folder,
                      leaf_input_dim=leaf_input_dim).to_client()


def main(args):
    # TODO: this is a really bad way to do this, for now it is acceptable
    global dataset_name, use_groupnorm, groupnorm_channels, train_loaders, val_loaders, test_loader, aux_loaders
    global leaf_input_dim
    global neve_active, neve_multiepoch, neve_multiepoch_epochs
    global neve_epsilon, neve_momentum, neve_alpha, neve_delta, neve_use_early_stop
    global scheduler_name, use_disk, model_name, use_pretrain, device, neve_only_last_layer, neve_use_lr_scheduler
    global base_lr, optimizer_name, momentum, weight_decay, amp
    global dataset_root
    global train
    global test
    global dataset_iid
    global num_clients
    global lda_concentration
    global seed
    global batch_size
    neve_epsilon = args.neve_epsilon
    neve_momentum = args.neve_momentum
    neve_alpha = args.neve_alpha
    neve_delta = args.neve_delta
    dataset_name = args.dataset_name.lower()
    use_groupnorm = args.model_use_groupnorm
    groupnorm_channels = args.model_groupnorm_groups
    use_pretrain = args.use_pretrain
    scheduler_name = args.scheduler_name
    use_disk = True
    base_lr = args.lr
    optimizer_name = args.optimizer
    momentum, weight_decay = args.momentum, args.weight_decay
    amp = args.amp
    neve_only_last_layer = args.neve_only_ll
    neve_use_lr_scheduler = args.neve_use_lr_scheduler
    neve_use_early_stop = args.neve_use_early_stop
    model_name = args.model_name
    neve_active = args.neve_active
    neve_multiepoch = args.neve_multiepoch
    neve_multiepoch_epochs = args.neve_multiepoch_epochs
    leaf_input_dim = args.leaf_input_dim
    dataset_root = args.dataset_root
    train = args.train
    test = args.test
    dataset_iid = args.dataset_iid
    num_clients = args.num_clients
    lda_concentration = args.lda_concentration
    seed = args.seed
    batch_size = args.batch_size
    # Cleanup neve_disk_folder
    if os.path.exists(disk_folder):
        shutil.rmtree(disk_folder)
    # Init seeds
    set_seeds(args.seed)
    # Initialize global model and data
    train, test, aux = get_dataset(args.dataset_root, args.dataset_name,
                                   aux_seed=args.seed, generate_aux_set=False)
    # Generate aux loaders
    aux_loaders = [load_aux_dataset(shape=(3, 24, 24), aux_seed=idx, batch_size=args.batch_size)
                   for idx in range(args.num_clients)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "cuda" else "cpu"
    print("Performing training on:", device)

    # Init wandb project
    wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, config=args, tags=args.wandb_tags)

    # Select strategy
    strategy_type = FedNeVeAvg if args.neve_active == "neve" else FedAvg
    if args.neve_active:
        strategy = strategy_type(
            client_sampler=get_client_sampler(args.clients_sampling_method,
                                              sampling_percentage=args.clients_sampling_percentage,
                                              sampling_wait_epochs=args.clients_sampling_wait_epochs,
                                              sampling_velocity_aging=args.clients_sampling_velocity_aging,
                                              sampling_highest_velocity=args.clients_sampling_highest_velocity,
                                              sampling_min_epochs=args.clients_sampling_min_epochs,
                                              sampling_use_probability=args.clients_sampling_use_probability),
            fit_metrics_aggregation_fn=weighted_average_fit,
            min_fit_clients=args.min_fit_clients,
            min_evaluate_clients=args.min_evaluate_clients,
            min_available_clients=args.num_clients,
            evaluate_metrics_aggregation_fn=weighted_average_eval
        )
    else:
        strategy = strategy_type(
            fit_metrics_aggregation_fn=weighted_average_fit,
            min_fit_clients=args.min_fit_clients,
            min_evaluate_clients=args.min_evaluate_clients,
            min_available_clients=args.num_clients,
            evaluate_metrics_aggregation_fn=weighted_average_eval
        )

    client_resources = {"num_cpus": 1, "num_gpus": 1 / args.num_clients}

    # Launch the simulation
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,  # A function to run a _virtual_ client when required
        num_clients=args.num_clients,  # Total number of clients available
        config=fl.server.ServerConfig(num_rounds=args.epochs),  # Specify number of FL rounds
        strategy=strategy,
        client_resources=client_resources,
    )
    # Save the model...

    # End wandb run
    wandb.run.finish()


if __name__ == "__main__":
    main(get_args("simulation"))
