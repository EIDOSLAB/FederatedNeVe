from arguments import get_args
from dataloaders import get_dataset, split_data


def main(args):
    train, test = get_dataset(args.dataset_root, args.dataset_name)
    train_loaders, val_loaders, test_loader = split_data(train, test, num_clients=1)
    # TODO ADD OPTIMIZERS
    # TODO ADD WANDB
    for epoch in range(0, args.epochs):
        print(f"Epoch [{epoch + 1}]/[{args.epochs}]:")
        # TODO TRAIN CYCLE


if __name__ == "__main__":
    main(get_args("classical"))
