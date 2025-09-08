import os
from collections import OrderedDict
from functools import reduce

import numpy as np
import torch
import wandb
from flwr.common import NDArrays

from my_federated.datasets.dataloader.loader import prepare_data, load_aux_dataset, load_transform, \
    get_dataset_fed_path, load_partition, load_loader, delete_dataset
from my_federated.models import get_model
from my_federated.my_flwr.clients import get_simulation_client
from my_federated.my_flwr.strategies.strategy_data import StrategyData
from my_federated.utils import set_seeds
from my_federated.utils.arguments import get_args
from my_federated.utils.trainer import eval_model


def weighted_avg(train_data: dict, key: str, size_key: str = 'size') -> float:
    """
    Compute the weighted average of a given value in the training results from a dictionary.

    Args:
        train_data (dict): Dictionary with client_id as keys and training results as values.
        key (str): The key inside 'train_results' for which to compute the weighted average.
        size_key (str): The key used for weights (usually 'size' or 'train_size').

    Returns:
        float: Weighted average of the requested value.
    """
    total_weight = 0
    weighted_sum = 0.0

    for client_id, entry in train_data.items():
        if "results" not in entry:
            continue
        result = entry['results']
        weight = result.get(size_key, 1)
        value = result.get(key, None)
        if value is not None:
            weighted_sum += value * weight
            total_weight += weight

    return weighted_sum / total_weight if total_weight > 0 else float('nan')


def aggregate(results: list[tuple[NDArrays, int]], scaling_factors: list[float]) -> NDArrays:
    """Compute weighted average."""

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * scaling_factors[idx] for layer in weights]
        for idx, (weights, _) in enumerate(results)
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates)
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


def update_model_parameters(model, parameters_aggregated):
    params_dict = list(zip(model.state_dict().keys(), parameters_aggregated))
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model


def main(args, eps: float = 1e-9):
    # Init wandb project
    wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, config=args, tags=args.wandb_tags)

    # Init seeds
    set_seeds(args.seed)

    # Prepare datasets
    print("\nPreparing datasets")
    if args.dataset_name.startswith("medmnist_"):
        args.dataset_root = os.path.join(args.dataset_root, "med_fl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "cuda" else "cpu"
    print("Performing training on:", device)

    (dataset_root, dataset_name, dataset_iid, num_clients, lda_concentration, seed, batch_size,
     use_groupnorm, groupnorm_channels, use_pretrain, model_name, base_lr, optimizer_name,
     scheduler_name, momentum, weight_decay, amp, neve_momentum, neve_only_last_layer,
     medmnist_size) = (
        args.dataset_root, args.dataset_name, args.dataset_iid, args.num_clients, args.lda_concentration,
        args.seed, args.batch_size, args.model_use_groupnorm, args.model_groupnorm_groups, args.use_pretrain,
        args.model_name, args.lr, args.optimizer, args.scheduler, args.momentum, args.weight_decay, args.amp,
        args.neve_momentum, args.neve_only_ll, args.medmnist_size)

    strategy_data = StrategyData(
        dataset_path=dataset_root, medmnist_size=medmnist_size, val_percentage=args.val_percentage,
        num_clients=num_clients, concentration=lda_concentration, dataset_iid=dataset_iid,
        seed=seed, strategy=args.strategy_name,
        dataset=dataset_name, model_name=model_name, device=device,
        use_pretrain=use_pretrain, amp=amp,
        use_groupnorm=use_groupnorm, groupnorm_channels=groupnorm_channels
    )

    try:
        train_distributions, num_classes, ds_task = prepare_data(args.dataset_root, args.dataset_name,
                                                                 num_clients=args.num_clients,
                                                                 seed=args.seed,
                                                                 split_iid=args.dataset_iid,
                                                                 concentration=args.lda_concentration,
                                                                 medmnist_size=args.medmnist_size,
                                                                 val_percentage=args.val_percentage,
                                                                 strategy_name=args.strategy_name,
                                                                 generate_distribution=True)
        strategy_data.dataset_task = ds_task

        dataset_fed_path = get_dataset_fed_path(strategy_data.dataset_path, strategy_data.dataset,
                                                medmnist_size=strategy_data.medmnist_size,
                                                val_size=strategy_data.val_percentage / 100, seed=strategy_data.seed,
                                                num_clients=strategy_data.num_clients,
                                                concentration=strategy_data.concentration,
                                                ds_iid=strategy_data.dataset_iid, strategy_name=strategy_data.strategy)
        test_partition = load_partition(dataset_fed_path, -1, partitions=["test"], transform_2_pil=True)["test"]
        test_partition.transform = load_transform(dataset_name, model_name)["test"]
        test_loader = load_loader(test_partition, batch_size=batch_size, shuffle=False)
        #
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

        test_client = get_simulation_client(args.dataset_root, args.dataset_name, aux_loader=aux_loaders[0],
                                            dataset_iid=args.dataset_iid, num_clients=args.num_clients,
                                            lda_concentration=args.lda_concentration,
                                            seed=args.seed, batch_size=args.batch_size,
                                            use_groupnorm=args.model_use_groupnorm,
                                            groupnorm_channels=args.model_groupnorm_groups,
                                            use_pretrain=args.use_pretrain,
                                            val_percentage=args.val_percentage,
                                            client_id=int(0), model_name=args.model_name, device=args.device,
                                            lr=args.lr,
                                            optimizer_name=args.optimizer, scheduler_name=args.scheduler,
                                            momentum=args.momentum, weight_decay=args.weight_decay, amp=args.amp,
                                            neve_momentum=args.neve_momentum,
                                            neve_only_last_layer=args.neve_only_ll,
                                            medmnist_size=args.medmnist_size,
                                            strategy_name=args.strategy_name,
                                            dataset_task=ds_task)

        test_model, _ = get_model(dataset=dataset_name, model_name=model_name,
                                  device=device, use_pretrain=use_pretrain,
                                  use_groupnorm=use_groupnorm,
                                  groupnorm_channels=groupnorm_channels)

        # Training cycle
        current_best_val_loss = float("inf")
        current_best_val_test_accuracy = 0.0
        current_best_epoch = -1
        clients = {}
        for client_id in range(args.num_clients):
            clients[client_id] = get_simulation_client(args.dataset_root, args.dataset_name,
                                                       aux_loader=aux_loaders[client_id],
                                                       dataset_iid=args.dataset_iid, num_clients=args.num_clients,
                                                       lda_concentration=args.lda_concentration,
                                                       seed=args.seed, batch_size=args.batch_size,
                                                       use_groupnorm=args.model_use_groupnorm,
                                                       groupnorm_channels=args.model_groupnorm_groups,
                                                       use_pretrain=args.use_pretrain,
                                                       client_id=int(client_id), model_name=args.model_name,
                                                       device=args.device,
                                                       lr=args.lr,
                                                       optimizer_name=args.optimizer, scheduler_name=args.scheduler,
                                                       momentum=args.momentum, weight_decay=args.weight_decay,
                                                       amp=args.amp,
                                                       neve_momentum=args.neve_momentum,
                                                       neve_only_last_layer=args.neve_only_ll,
                                                       velocity_stop_threshold=args.neve_velocity_stop_threshold,
                                                       max_idle_epochs=args.neve_max_idle_epochs,
                                                       medmnist_size=args.medmnist_size,
                                                       strategy_name=args.strategy_name,
                                                       dataset_task=ds_task,
                                                       pin_data_in_memory=True)

        for epoch in range(0, args.epochs):
            models_parameters = {}
            train_data = {}
            eval_data = {}
            server_parameters = test_client._get_model_parameters(test_model)
            for client_id in range(args.num_clients):
                client = clients[client_id]

                model_parameters, train_size, train_results = client.fit(server_parameters, {"round": epoch})
                eval_loss, eval_size, eval_results = client.evaluate(server_parameters, {"round": epoch})

                train_data[client_id] = {
                    "performed_training": train_results.get("performed_training", False)
                }
                eval_data[client_id] = {
                    "eval_loss": eval_loss,
                    "eval_size": eval_size,
                    "results": eval_results,
                }
                if train_results.get("performed_training", False):
                    models_parameters[client_id] = (model_parameters, train_size)
                    train_data[client_id]["train_size"] = train_size
                    train_data[client_id]["results"] = train_results
            # End clients cycle

            if not models_parameters:
                wandb.log({"neve_early_stop": epoch})
                break
            ##
            ##
            ##
            # Calculate the total number of examples used during training
            # By default we use the number of samples to define the scaling factor of a client
            scaling_factors = [client_training_data["train_size"]  for _, client_training_data
                               in train_data.items() if "train_size" in client_training_data]
            # Get total number of examples used in this training cycle
            num_examples_total = sum(scaling_factors)
            # Normalize scaling factors
            scaling_factors = [scaling_factor / (num_examples_total + eps) for scaling_factor in scaling_factors]
            #
            merged_parameters = aggregate([(val[0], val[1]) for _, val in models_parameters.items()],
                                          scaling_factors=scaling_factors)
            #
            test_model = update_model_parameters(test_model, merged_parameters)
            ##
            ##
            ##
            test_stats = eval_model(test_model, test_loader, ds_task, device, amp,
                                    epoch=epoch, run_type="Test")
            test_loss, test_acc_1 = test_stats["loss"], test_stats["accuracy"]["top1"]

            train_acc_t1 = weighted_avg(train_data, key="accuracy_top1")
            train_loss = weighted_avg(train_data, key="loss")
            lr = weighted_avg(train_data, key="lr")
            val_acc_t1 = weighted_avg(train_data, key="accuracy_top1")
            val_loss = weighted_avg(train_data, key="loss")
            if val_loss < current_best_val_loss:
                current_best_val_loss = val_loss
                current_best_val_test_accuracy = test_acc_1
                current_best_epoch = epoch
            wandb_logs = {
                "train": {
                    "accuracy": {
                        "top1": train_acc_t1,
                    },
                    "loss": train_loss
                },
                "val": {
                    "accuracy": {
                        "top1": val_acc_t1,
                    },
                    "loss": val_loss,
                },
                "test": {
                    "accuracy": {
                        "top1": float(test_acc_1),
                    },
                    "loss": test_loss,
                },
                "best_on_val": {
                    "best_loss": current_best_val_loss,
                    "best_test_accuracy": current_best_val_test_accuracy,
                    "best_epoch": current_best_epoch,
                },
                "lr": {
                    "0": lr
                },
                "aux": {},
            }
            for client_id, client_data in train_data.items():
                if client_data.get("performed_training", False):
                    wandb_logs["aux"][client_id] = {key: val for key, val in client_data.get("results", {}).items() if
                                                    key.startswith('neve.')}
                    wandb_logs["aux"][client_id]["performed_training"] = 1
                else:
                    wandb_logs["aux"][client_id] = {"performed_training": 0}
            #
            print("\n-----")
            print(f"Epoch [{epoch + 1}]/[{args.epochs}]:")
            print(f"LR: {wandb_logs['lr']}")
            print(f"Train: {wandb_logs['train']}")
            print(f"Val: {wandb_logs['val']}")
            print(f"Test: {wandb_logs['test']}")
            print(f"NeVe - Aux: avg.vel. {wandb_logs['aux']} ")
            print("-----\n")

            # Log on wandb project
            wandb.log(wandb_logs)
    except Exception as e:
        print(f"federated.py -> Error: {e}")
    finally:
        # Cleanup data
        delete_dataset(dataset_root, dataset_name, dataset_iid, args.val_percentage, num_clients, seed, medmnist_size,
                       args.strategy_name)

        # Save model...

        # End wandb run
        wandb.run.finish()


if __name__ == "__main__":
    main(get_args("federated"))
