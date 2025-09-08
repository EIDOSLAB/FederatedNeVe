import wandb

from my_federated.datasets.dataloader.loader import prepare_data, load_aux_dataset, load_transform
from my_federated.my_flwr.clients import get_simulation_client
from my_federated.utils import set_seeds
from my_federated.utils.arguments import get_args


def main(args):
    # Init seeds
    set_seeds(args.seed)

    train_distributions, num_classes, ds_task = prepare_data(args.dataset_root, args.dataset_name,
                                                             num_clients=args.num_clients,
                                                             seed=args.seed,
                                                             split_iid=args.dataset_iid,
                                                             concentration=args.lda_concentration,
                                                             medmnist_size=args.medmnist_size,
                                                             generate_distribution=True)

    aux_loader_transform = load_transform(args.dataset_name, args.model_name, aux_transform=True)
    if "chestmnist" in args.dataset_name:
        aux_loaders = [load_aux_dataset(shape=(3, 32, 32), aux_seed=args.seed, number_samples=10,
                                        batch_size=args.batch_size, transform=aux_loader_transform,
                                        labels_size=num_classes)]
    else:
        aux_loaders = [load_aux_dataset(shape=(3, 32, 32), aux_seed=args.seed, number_samples=10,
                                        batch_size=args.batch_size, transform=aux_loader_transform,
                                        labels_size=1)]

    client = get_simulation_client(args.dataset_root, args.dataset_name, aux_loader=aux_loaders[0],
                                   dataset_iid=args.dataset_iid, num_clients=args.num_clients,
                                   lda_concentration=args.lda_concentration,
                                   seed=args.seed, batch_size=args.batch_size,
                                   use_groupnorm=args.model_use_groupnorm,
                                   groupnorm_channels=args.model_groupnorm_groups,
                                   use_pretrain=args.use_pretrain,
                                   client_id=int(0), model_name=args.model_name, device=args.device, lr=args.lr,
                                   optimizer_name=args.optimizer, scheduler_name=args.scheduler,
                                   momentum=args.momentum, weight_decay=args.weight_decay, amp=args.amp,
                                   neve_momentum=args.neve_momentum,
                                   neve_only_last_layer=args.neve_only_ll,
                                   medmnist_size=args.medmnist_size,
                                   strategy_name=args.strategy_name,
                                   dataset_task=ds_task)

    # Init seeds
    set_seeds(args.seed)  # Just to be sure to be in the same spot whatever we generated the aux dataset or not

    # Init wandb project
    wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, config=args, tags=args.wandb_tags)

    # Training cycle
    model_parameters = client.get_parameters({})
    for epoch in range(0, args.epochs):
        model_parameters, _, train_results = client.fit(model_parameters, {})
        _, _, eval_results = client.evaluate(model_parameters, {})
        print()
        wandb_logs = {
            "train": {
                "accuracy": {
                    "top1": train_results["train_accuracy_top1"],
                },
                # "balanced_accuracy": {
                #    "top1": train_results["balanced_accuracy_top1"],
                # },
                # "auc": train_results["auc"],
                "loss": train_results["train_loss"]
            },
            "val": {
                "accuracy": {
                    "top1": eval_results["val_accuracy_top1"],
                },
                # "balanced_accuracy": {
                #    "top1": eval_results["val_balanced_accuracy_top1"],
                # },
                # "auc": eval_results["val_auc"],
                "loss": eval_results["val_loss"],
            },
            "test": {
                "accuracy": {
                    "top1": eval_results["test_accuracy_top1"],
                },
                # "balanced_accuracy": {
                #    "top1": eval_results["test_balanced_accuracy_top1"],
                # },
                # "auc": eval_results["test_auc"],
                "loss": eval_results["test_loss"],
            },
            "lr": {
                "0": train_results["lr"]
            },
            "aux": {key: val for key, val in train_results.items() if key.startswith('neve.')},
        }
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

    # Save model...

    # End wandb run
    wandb.run.finish()


if __name__ == "__main__":
    main(get_args("classical"))
