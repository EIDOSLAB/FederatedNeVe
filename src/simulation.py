import os.path
import sys

# Verifica il sistema operativo
if sys.platform.startswith('win'):
    import msvcrt
else:
    import fcntl

import flwr as fl
import torch
import wandb
from flwr.common import Context
import matplotlib.pyplot as plt

from my_federated.my_flwr.strategies.strategy_data import StrategyData
from my_federated.my_flwr.clients import get_simulation_client
from my_federated.utils.arguments import get_args
from my_federated.datasets.dataloader.loader import load_aux_dataset, prepare_data, load_transform, delete_dataset
from my_federated.my_flwr.aggregation.avg_aggregate import weighted_average_fit, weighted_average_eval
from my_federated.my_flwr.strategies.FedAvgWConfig import FedAvgWConfig, custom_on_fit_config_fn
from my_federated.utils import set_seeds
from my_federated.NeVe.strategy.fed_neve_avg import FedNeVeAvg

aux_loaders = []


def get_client_id(cid_file_path: str = "client_id.txt", num_clients: int = 10):
    cid = 0
    # Apri il file in modalità lettura/scrittura
    with open(cid_file_path, "r+") as f:
        # Ottieni il lock sul file (dipende dal sistema operativo)
        if sys.platform.startswith('win'):  # Windows
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 0)  # Lock esclusivo
        else:  # Linux/macOS
            fcntl.flock(f, fcntl.LOCK_EX)  # LOCK_EX per un lock esclusivo
        # Se il file esiste, leggi il valore e incrementalo
        if os.path.exists(cid_file_path):
            f.seek(0)  # Posiziona il cursore all'inizio del file
            contenuto = f.read().strip()
            cid = int(contenuto) + 1

        # Assicurati che cid sia all'interno del range [0, num_clients)
        cid = cid % num_clients

        # Posiziona di nuovo il cursore all'inizio per scrivere il nuovo valore
        f.seek(0)
        f.truncate()  # Rimuove il contenuto precedente
        f.write(str(cid))  # Scrivi il nuovo valore

    return cid


def get_client_fn(dataset_root, dataset_name, dataset_iid, num_clients, lda_concentration, seed, batch_size,
                  use_groupnorm, groupnorm_channels, use_pretrain, model_name, device, base_lr, optimizer_name,
                  scheduler_name,
                  momentum, weight_decay, amp, neve_momentum, neve_only_last_layer, medmnist_size,
                  strategy: str = "fedavg", train_distributions: dict = {},
                  val_percentage: int = 10, dataset_task: str = "multi-class"):
    def client_fn(context: Context):
        cid = get_client_id(num_clients=num_clients)
        aux_loader = aux_loaders[int(cid) % len(aux_loaders)] if aux_loaders else None
        print("Created client with cid:", cid)
        return get_simulation_client(dataset_root, dataset_name, aux_loader=aux_loader,
                                     dataset_iid=dataset_iid, num_clients=num_clients,
                                     lda_concentration=lda_concentration,
                                     seed=seed, batch_size=batch_size,
                                     use_groupnorm=use_groupnorm, groupnorm_channels=groupnorm_channels,
                                     use_pretrain=use_pretrain,
                                     client_id=int(cid), model_name=model_name, device=device, lr=base_lr,
                                     optimizer_name=optimizer_name, scheduler_name=scheduler_name,
                                     momentum=momentum, weight_decay=weight_decay, amp=amp,
                                     neve_momentum=neve_momentum,
                                     neve_only_last_layer=neve_only_last_layer,
                                     medmnist_size=medmnist_size,
                                     strategy_name=strategy,
                                     data_distribution=train_distributions.get(int(cid)),
                                     val_percentage=val_percentage,
                                     dataset_task=dataset_task).to_client()

    #
    return client_fn


def main(args):
    global aux_loaders
    # Init seeds
    set_seeds(args.seed)

    # Prepare datasets
    print("\nPreparing datasets")
    if args.dataset_name.startswith("medmnist_"):
        args.dataset_root = os.path.join(args.dataset_root, "med_fl")

    train_distributions, num_classes, ds_task = prepare_data(args.dataset_root, args.dataset_name,
                                                             num_clients=args.num_clients,
                                                             seed=args.seed,
                                                             split_iid=args.dataset_iid,
                                                             concentration=args.lda_concentration,
                                                             medmnist_size=args.medmnist_size,
                                                             val_percentage=args.val_percentage,
                                                             strategy_name=args.strategy_name,
                                                             generate_distribution=True)

    with open("client_id.txt", "w") as f:
        f.write(str(-1))
    print("\nDatasets preparation complete.\n")
    # Generate aux loaders
    aux_loader_transform = load_transform(args.dataset_name, args.model_name, aux_transform=True)
    if ds_task == "multi-label, binary-class":
        aux_loaders = [
            load_aux_dataset(shape=(3, 32, 32), aux_seed=idx, batch_size=args.batch_size,
                             transform=aux_loader_transform, labels_size=num_classes)
            for idx in range(args.num_clients)]
    else:
        aux_loaders = [
            load_aux_dataset(shape=(3, 32, 32), aux_seed=idx, batch_size=args.batch_size,
                             transform=aux_loader_transform, labels_size=1)
            for idx in range(args.num_clients)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "cuda" else "cpu"
    print("Performing training on:", device)

    # Init wandb project
    if args.wandb_project_name == "NeVe-Federated-Strategy":
        args.wandb_project_name = "NeVe-Federated-Strategy-Simulation"
    wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, config=args, tags=args.wandb_tags)

    if args.print_clients_distribution and train_distributions:
        print("Generating datasets distributions")
        distribution_logs = {"clients": {}}
        for idx, (cid, distribution) in enumerate(train_distributions.items()):
            print(f"Generating distribution for dataset [{idx}/{len(train_distributions)}]")
            classes = [class_id for class_id in range(len(distribution))]
            plt.figure(figsize=(12, 6))
            plt.bar(classes, distribution, color='skyblue')
            plt.xlabel('Classes')
            plt.ylabel('Instances [%]')
            plt.title('Instances/Classes Distribution')
            # Creazione del grafico a barre con matplotlib
            if cid not in distribution_logs["clients"].keys():
                distribution_logs["clients"][cid] = {"data_distribution": None}
            distribution_logs["clients"][cid]["data_distribution"] = wandb.Image(plt,
                                                                                 caption="Instances/Classes Distribution")
            plt.close()
        wandb.log(distribution_logs, commit=False)

    print("Preparing strategy and system resources")
    (dataset_root, dataset_name, dataset_iid, num_clients, lda_concentration, seed, batch_size,
     use_groupnorm, groupnorm_channels, use_pretrain, model_name, device, base_lr, optimizer_name,
     scheduler_name, momentum, weight_decay, amp, neve_momentum, neve_only_last_layer,
     medmnist_size) = (
        args.dataset_root, args.dataset_name, args.dataset_iid, args.num_clients, args.lda_concentration,
        args.seed, args.batch_size, args.model_use_groupnorm, args.model_groupnorm_groups, args.use_pretrain,
        args.model_name,
        device, args.lr, args.optimizer, args.scheduler, args.momentum, args.weight_decay, args.amp,
        args.neve_momentum, args.neve_only_ll, args.medmnist_size)

    strategy_data = StrategyData(
        dataset_path=dataset_root, medmnist_size=medmnist_size, val_percentage=args.val_percentage,
        num_clients=num_clients, concentration=lda_concentration, dataset_iid=dataset_iid,
        seed=seed, strategy=args.strategy_name, dataset_task=ds_task,
        dataset=dataset_name, model_name=model_name, device=device,
        use_pretrain=use_pretrain, amp=amp,
        use_groupnorm=use_groupnorm, groupnorm_channels=groupnorm_channels
    )
    # Select strategy
    match args.strategy_name:
        case "neve":
            strategy = FedNeVeAvg(
                on_fit_config_fn=custom_on_fit_config_fn,
                inverse_velocity=args.neve_velocity_inverted,
                temperature=args.neve_softmax_temperature,
                epochs=args.epochs,
                strategy_data=strategy_data,
                velocity_aggregation_fn=args.neve_velocity_aggregation_fn,
                fit_metrics_aggregation_fn=weighted_average_fit,
                min_fit_clients=args.min_fit_clients,
                min_evaluate_clients=args.min_evaluate_clients,
                min_available_clients=min(args.num_clients, args.min_fit_clients, args.min_evaluate_clients),
                evaluate_metrics_aggregation_fn=weighted_average_eval
            )
        # By default, we fall off to FedAvg
        case _:
            strategy = FedAvgWConfig(
                fit_metrics_aggregation_fn=weighted_average_fit,
                strategy_data=strategy_data,
                on_fit_config_fn=custom_on_fit_config_fn,
                min_fit_clients=args.min_fit_clients,
                min_evaluate_clients=args.min_evaluate_clients,
                min_available_clients=min(args.num_clients, args.min_fit_clients, args.min_evaluate_clients),
                evaluate_metrics_aggregation_fn=weighted_average_eval
            )

    num_cpus = args.max_core_count / args.num_clients
    num_cpus = num_cpus if num_cpus > 0 else 1
    client_resources = {"num_cpus": num_cpus, "num_gpus": torch.cuda.device_count() / args.num_clients}

    print("Starting the simulation")

    # Launch the simulation
    hist = fl.simulation.start_simulation(
        client_fn=get_client_fn(dataset_root, dataset_name, dataset_iid, num_clients, lda_concentration, seed,
                                batch_size,
                                use_groupnorm, groupnorm_channels, use_pretrain, model_name, device, base_lr,
                                optimizer_name, scheduler_name,
                                momentum, weight_decay, amp, neve_momentum, neve_only_last_layer,
                                medmnist_size, strategy=args.strategy_name, train_distributions=train_distributions,
                                dataset_task=ds_task),
        # A function to run a _virtual_ client when required
        num_clients=args.num_clients,  # Total number of clients available
        config=fl.server.ServerConfig(num_rounds=args.epochs),  # Specify number of FL rounds
        strategy=strategy,
        client_resources=client_resources,
    )

    # Cleanup data
    delete_dataset(dataset_root, dataset_name, dataset_iid, 10, num_clients, seed, medmnist_size,
                   args.strategy_name)
    # Save the model...

    # End wandb run
    wandb.run.finish()


if __name__ == "__main__":
    main(get_args("federated"))
