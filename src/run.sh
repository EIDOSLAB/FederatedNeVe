#!/bin/bash

# Parametri per il server ed i clients
params="--amp 1 --device cuda --batch-size 100 --epochs 250 --lr 0.001"
optimizer="--optimizer sgd"
scheduler="--scheduler-name baseline"
model_use_groupnorm="--model-use-groupnorm 1"
dataset="--dataset-name cifar10"
model="--model-name resnet18"

# Numero di clients da avviare
clients="--num-clients 10"
min_fit_clients="--min-fit-clients 10"
min_eval_clients="--min-evaluate-clients 10"

clients_sampling_method="--clients-sampling-method velocity"
clients_sampling_percentage="--clients-sampling-percentage 0.5"
clients_sampling_velocity_aging="--clients-sampling-velocity-aging 0.1"

# Dataset split distribution
dataset_iid="--dataset-iid 0"
lda_concentration="--lda-concentration 0.1"

# Use Neve
use_neve="--neve-active 1"
neve_use_lr_scheduler="--neve-use-lr-scheduler 0"
neve_only_ll="--neve-only-ll 1"

wandb_tags="--wandb-tags SAMPLER-VELOCITY_0.5"

number_of_seeds=3
single_gpu=1
num_clients=10

echo "Preparazione Training Federato in modalità Batch"
# Loop per le run con seed differente
for ((current_seed=0; current_seed<number_of_seeds; current_seed+=1)); do
    seed="--seed $current_seed"
    echo "Batch Federato con seed: [$current_seed / $number_of_seeds]"

    # Definisci i parametri comuni
    common_params="$params $clients $optimizer $scheduler $use_neve $neve_only_ll $neve_use_lr_scheduler $model_use_groupnorm $seed $dataset $model $dataset_iid $lda_concentration  $clients_sampling_method $clients_sampling_percentage $clients_sampling_velocity_aging"

    # Avvia il server
    echo "Avvio del server con parametri: $common_params $min_fit_clients $min_eval_clients $wandb_tags"
    python server.py $common_params $min_fit_clients $min_eval_clients $wandb_tags &

    # Ottieni l'ID del processo del server
    server_pid=$!

    # Aspetta che il server parta prima di far partire i processi figli (10 secondi dovrebbero bastare)
    sleep 10

    # Loop per avviare i clients
    for ((i=0; i<num_clients; i+=1)); do
        # Se siamo su singola GPU imposto a 0
        if [ $single_gpu -eq 1 ]; then
            gpu_id=0
        else
            # Distribuisco i clients fra le 2 GPU
            if ((i % 2 == 0)); then
                gpu_id=0
            else
                gpu_id=1
            fi
        fi

        echo "Avvio del client $i con parametri: $common_params --current-client $i, sulla GPU: $gpu_id"
        CUDA_VISIBLE_DEVICES=$gpu_id python client.py $common_params --current-client $i &
    done

    # Attendi che tutti i processi dei client siano completati
    wait

    # Attendi che il server termini autonomamente
    wait $server_pid

    sleep 10

    echo "Terminazione del server"
done

echo "Terminazione Batch di Trainings"
