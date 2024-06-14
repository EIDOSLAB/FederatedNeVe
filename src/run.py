import platform
import subprocess
# ----- ----- ----- ----- -----
# TODO: FIX SRC IMPORTS IN A BETTER WAY
import sys
import time
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# ----- ----- ----- ----- -----
from src.dataloaders import get_dataset, prepare_data

# Basic params
amp = 1
device = "cuda"
batch_size = 100
epochs = 250
lr = 0.001

# Model and relative utils params
model = "resnet18"
model_use_groupnorm = 1
optimizer = "sgd"
scheduler = "baseline"

# Data params
dataset_root = "../datasets"
dataset_name = "cifar10"
dataset_iid = 0
lda_concentration = 0.1

# NeVe params
use_neve = 1
neve_use_lr_scheduler = 0
neve_only_ll = 1

# Clients params
clients = 10
min_fit_clients = 10
min_eval_clients = 10

# Clients sampling params
clients_sampling_method = "velocity"
clients_sampling_percentage = 0.5
clients_sampling_velocity_aging = 0.1
clients_sampling_highest_velocity = 1
clients_sampling_wait_epochs = 10

# Simulation params
number_of_seeds = 3
single_gpu = 1
num_clients = 10


def _prepare_dataset(seed: int):
    print("Preparazione dataset per il training federato...")
    split_iid = True if dataset_iid == 1 else False
    # Load data
    train, test, aux = get_dataset(dataset_root, dataset_name)
    _ = prepare_data(dataset_root, dataset_name, train, test, aux,
                     split_iid=split_iid, num_clients=num_clients,
                     concentration=lda_concentration, seed=seed, batch_size=batch_size)
    print("Preparazione dataset per il training federato completata.")


def _start_simulation(seed: int):
    # Parameters for server and clients
    basic_params = f"--amp {str(amp)} --device {device} --batch-size {str(batch_size)} " \
                   f"--epochs {str(epochs)} --lr {str(lr)} --seed {str(seed)}"
    data_params = f"--dataset-root {dataset_root} --dataset-name {dataset_name} --dataset-iid {str(dataset_iid)} " \
                  f"--lda-concentration {str(lda_concentration)}"
    model_params = f"--optimizer {optimizer} --scheduler-name {scheduler} --model-name {model} " \
                   f"--model-use-groupnorm {str(model_use_groupnorm)}"
    neve_params = f"--neve-active {str(use_neve)} --neve-only-ll {str(neve_only_ll)} " \
                  f"--neve-use-lr-scheduler {str(neve_use_lr_scheduler)}"
    clients_params = f"--num-clients {str(num_clients)} --min-fit-clients {str(min_fit_clients)} " \
                     f"--min-evaluate-clients  {str(min_eval_clients)}"
    sampling_params = f"--clients-sampling-method {clients_sampling_method} " \
                      f"--clients-sampling-percentage {str(clients_sampling_percentage)} " \
                      f"--clients-sampling-wait-epochs {str(clients_sampling_wait_epochs)} " \
                      f"--clients-sampling-velocity-aging {str(clients_sampling_velocity_aging)} " \
                      f"--clients-sampling-highest-velocity {str(clients_sampling_highest_velocity)}" \

    # Common params between server and client
    common_params = f"{basic_params} {data_params} {model_params} {neve_params} {clients_params} {sampling_params}"

    # Creation of wandb tags
    wandb_tags = f"--wandb-tags SAMPLER-{clients_sampling_method.upper()}_{clients_sampling_percentage}"
    if clients_sampling_method.lower() == "velocity":
        wandb_tags += f"_AGING_{clients_sampling_velocity_aging}"
        wandb_tags += " SAMPLE-VELOCITY_"
        if clients_sampling_highest_velocity == 1:
            wandb_tags += "HIGHEST"
        else:
            wandb_tags += "LOWEST"
    if dataset_iid == 0:
        wandb_tags += f" NON-IID LDA-CONCENTRATION_{str(lda_concentration)}"
    else:
        wandb_tags += " IID"
    wandb_tags += f" WAIT-EPOCHS_{str(clients_sampling_wait_epochs)}"

    # Start the server
    server_command = f"python server.py {common_params} {wandb_tags}"
    print(f"Avvio del server: {server_command}")
    server_process = subprocess.Popen(server_command, shell=True)
    # Wait some time so that the server can start
    time.sleep(10)

    # Loop to start all clients
    client_processes = []
    for i in range(num_clients):
        # If we have a single gpu set the gpu_id to 0 for all clients
        if single_gpu == 1:
            gpu_id = 0
        # Otherwise, split clients between the 2 gpus
        else:
            gpu_id = 0 if i % 2 == 0 else 1

        if platform.system() == 'Linux':
            client_command = f"CUDA_VISIBLE_DEVICES={gpu_id} python client.py {common_params} --current-client {i}"
        else:
            client_command = f"python client.py {common_params} --current-client {i}"

        print(f"Avvio del client {i}: {client_command}, sulla GPU: {gpu_id}")
        client_process = subprocess.Popen(client_command, shell=True)
        client_processes.append(client_process)

    # Wait for all clients to finish
    for client_process in client_processes:
        client_process.wait()

    # Wait for the server to finish
    server_process.wait()
    print("Terminazione del server")


def main():
    print("Preparazione Training Federato in modalità Batch")
    for current_seed in range(number_of_seeds):
        print(f"Batch Federato con seed: [{current_seed} / {number_of_seeds}]")
        _prepare_dataset(current_seed)
        _start_simulation(current_seed)
        # Wait a bit before starting the next simulation so that everything has time to close
        time.sleep(10)
    print("Terminazione Training Federato in modalità Batch")


if __name__ == "__main__":
    main()
