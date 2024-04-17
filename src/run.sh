#!/bin/bash

# Parametri per il server ed i clients
params="--amp 1 --device cuda --batch-size 100 --epochs 250 --lr 0.01"
optimizer="--optimizer sgd"
scheduler="--scheduler-name neve"
neve_only_ll="--neve-only-ll 1"
model_use_groupnorm="--model-use-groupnorm 1"
seed="--seed 0"
dataset="--dataset-name cifar10"
model="--model-name resnet18"

# Numero di clients da avviare
num_clients=20
clients="--num-clients $num_clients"

# Avvia il server
echo "Avvio del server con parametri: $params $clients $optimizer $scheduler $neve_only_ll $model_use_groupnorm $seed $dataset $model"
python server.py $params $clients $optimizer $scheduler $neve_only_ll $model_use_groupnorm $seed $dataset $model &

# Ottieni l'ID del processo del server
server_pid=$!

# Aspettiamo che il server parta prima di far partire i processi figli (10 secondi dovrebbero bastare)
sleep 10

# Loop per avviare i clients
for ((i=0; i<num_clients; i+=1))
do
    # Determina se i è pari o dispari
    is_even=$((i % 2 == 0))

    # Imposta CUDA_VISIBLE_DEVICES in base a is_even
    if [ $is_even -eq 1 ]; then
        gpu_id=0
    else
        gpu_id=1
    fi
    echo "Avvio del client $i con parametri: $params $clients $optimizer $scheduler $neve_only_ll $model_use_groupnorm $seed $dataset $model --current-client $i, sulla GPU: $gpu_id"
    CUDA_VISIBLE_DEVICES=$gpu_id python client.py $params $clients $optimizer $scheduler $neve_only_ll $model_use_groupnorm $seed $dataset $model "--current-client" $i &
done

# Attendi che tutti i processi dei client siano completati
wait

# Attendi che il server termini autonomamente
wait $server_pid

echo "Terminazione del server"
