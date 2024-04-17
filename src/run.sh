#!/bin/bash

# Parametri per il server ed i clients
params="--amp 1 --device cuda --batch-size 100 --epochs 250 --lr 0.01"
optimizer="--optimizer sgd"
scheduler="--scheduler-name neve"
neve_only_ll="--neve-only-ll 1"
model_use_groupnorm="--model-use-groupnorm 1"
dataset="--dataset-name cifar10"
model="--model-name resnet18"

# Numero di clients da avviare
num_clients=10
clients="--num-clients $num_clients"

number_of_seeds=10
single_gpu=1

echo "Preparazione Training Federato in modalità Batch"
# Loop per le run con seed differente
for ((current_seed=0; current_seed<number_of_seeds; current_seed+=1))
do
    seed="--seed $current_seed"
    # Avvia il server
    echo "Batch Federato con seed: [$current_seed / $number_of_seeds]"
    echo "Avvio del server con parametri: $params $clients $optimizer $scheduler $neve_only_ll $model_use_groupnorm $seed $dataset $model"
    python server.py $params $clients $optimizer $scheduler $neve_only_ll $model_use_groupnorm $seed $dataset $model &

    # Ottieni l'ID del processo del server
    server_pid=$!

    # Aspettiamo che il server parta prima di far partire i processi figli (10 secondi dovrebbero bastare)
    sleep 10

    # Loop per avviare i clients
    for ((i=0; i<num_clients; i+=1))
    do
        # Se siamo singola gpu imposto a 0
        if [ $single_gpu -eq 1]; then
            gpu_id=0
        # Altrimenti distribuisco i clients fra le 2 GPU
        else
            # Determina se i è pari o dispari
            is_even=$((i % 2 == 0))
            # Imposta CUDA_VISIBLE_DEVICES in base a is_even
            if [ $is_even -eq 1 ]; then
                gpu_id=0
            else
                gpu_id=1
            fi
        fi

        echo "Avvio del client $i con parametri: $params $clients $optimizer $scheduler $neve_only_ll $model_use_groupnorm $seed $dataset $model --current-client $i, sulla GPU: $gpu_id"
        CUDA_VISIBLE_DEVICES=$gpu_id python client.py $params $clients $optimizer $scheduler $neve_only_ll $model_use_groupnorm $seed $dataset $model "--current-client" $i &
    done

    # Attendi che tutti i processi dei client siano completati
    wait

    # Attendi che il server termini autonomamente
    wait $server_pid

    sleep 10
    echo "Terminazione del server"
done

echo "Terminazione Batch di Trainings"

