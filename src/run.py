import platform
import subprocess
import time

# Parametri per il server ed i clients
params = "--amp 1 --device cuda --batch-size 100 --epochs 250 --lr 0.001".strip()
optimizer = "--optimizer sgd".strip()
scheduler = "--scheduler-name baseline".strip()
model_use_groupnorm = "--model-use-groupnorm 1".strip()
dataset = "--dataset-name cifar10".strip()
model = "--model-name resnet18".strip()

# Numero di clients da avviare
clients = "--num-clients 10".strip()
min_fit_clients = "--min-fit-clients 5".strip()
min_eval_clients = "--min-evaluate-clients 5".strip()

clients_sampling_method = "--clients-sampling-method velocity".strip()
clients_sampling_percentage = "--clients-sampling-percentage 0.5".strip()
clients_sampling_velocity_aging = "--clients-sampling-velocity-aging 0.1".strip()

# Dataset split distribution
dataset_iid = "--dataset-iid 0".strip()
lda_concentration = "--lda-concentration 0.1".strip()

# Use Neve
use_neve = "--neve-active 0".strip()
neve_use_lr_scheduler = "--neve-use-lr-scheduler 0".strip()
neve_only_ll = "--neve-only-ll 1".strip()

wandb_tags = f"--wandb-tags SAMPLER-{clients_sampling_method.split(' ')[-1].upper()}" \
             f"_{clients_sampling_percentage.split(' ')[-1]}"

if clients_sampling_method.split(" ")[-1] == "velocity":
    wandb_tags += f"_AGING_{clients_sampling_velocity_aging.split(' ')[-1]}"

if dataset_iid.split(" ")[-1] == "0":
    wandb_tags += f" NON-IID LDA-CONCENTRATION_{lda_concentration.split(' ')[-1]}"
else:
    wandb_tags += " IID"

number_of_seeds = 3
single_gpu = 1
num_clients = 10

print("Preparazione Training Federato in modalità Batch")

# Loop per le run con seed differente
for current_seed in range(number_of_seeds):
    seed = f"--seed {current_seed}"
    print(f"Batch Federato con seed: [{current_seed} / {number_of_seeds}]")

    # Definisci i parametri comuni
    common_params = f"{params} {clients} {optimizer} {scheduler} {use_neve} {neve_only_ll} {neve_use_lr_scheduler} " \
                    f"{model_use_groupnorm} {seed} {dataset} {model} {dataset_iid} {lda_concentration} " \
                    f"{clients_sampling_method} {clients_sampling_percentage} {clients_sampling_velocity_aging}"

    # Avvia il server
    server_command = f"python server.py {common_params} {min_fit_clients} {min_eval_clients} {wandb_tags}"
    print(f"Avvio del server con parametri: {server_command}")
    server_process = subprocess.Popen(server_command, shell=True)

    # Aspetta che il server parta prima di far partire i processi figli (10 secondi dovrebbero bastare)
    time.sleep(10)

    # Loop per avviare i clients
    client_processes = []
    for i in range(num_clients):
        # Se siamo su singola GPU imposto a 0
        if single_gpu == 1:
            gpu_id = 0
        else:
            # Distribuisco i clients fra le 2 GPU
            gpu_id = 0 if i % 2 == 0 else 1

        if platform.system() == 'Linux':
            client_command = f"CUDA_VISIBLE_DEVICES={gpu_id} python client.py {common_params} --current-client {i}"
        else:
            client_command = f"python client.py {common_params} --current-client {i}"

        print(f"Avvio del client {i} con parametri: {client_command}, sulla GPU: {gpu_id}")
        client_process = subprocess.Popen(client_command, shell=True)
        client_processes.append(client_process)

    # Attendi che tutti i processi dei client siano completati
    for client_process in client_processes:
        client_process.wait()

    # Attendi che il server termini autonomamente
    server_process.wait()

    time.sleep(10)
    print("Terminazione del server")

print("Terminazione Batch di Trainings")
