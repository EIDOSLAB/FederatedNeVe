import wandb

from arguments import get_args
from dataloaders import get_dataset, split_data
from utils import set_seeds


def main(args):
    # Init seeds
    set_seeds(args.seed)
    # Load Data
    train, test = get_dataset(args.dataset_root, args.dataset_name)
    train_loaders, val_loaders, test_loader = split_data(train, test, num_clients=1)
    train_loader, val_loader = train_loaders[0], val_loaders[0]

    # TODO: Load Optimizers and model

    # Init wandb project
    wandb.init(project=args.project_name, name=args.run_name, config=args)

    # Training cycle
    for epoch in range(0, args.epochs):
        print(f"Epoch [{epoch + 1}]/[{args.epochs}]:")
        # TODO TRAIN CYCLE
    # Save model...

    # End wandb run
    wandb.run.finish()


if __name__ == "__main__":
    main(get_args("classical"))
