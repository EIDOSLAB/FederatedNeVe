import torch
import wandb

from arguments import get_args
from dataloaders import get_dataset, split_data
from models import get_model
from utils import set_seeds, get_optimizer, get_scheduler
from utils.trainer import train_epoch


def main(args):
    # Init seeds
    set_seeds(args.seed)
    # Load Data
    train, test = get_dataset(args.dataset_root, args.dataset_name)
    train_loaders, val_loaders, test_loader = split_data(train, test, num_clients=1)
    train_loader, val_loader = train_loaders[0], val_loaders[0]
    data_loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "aux": None,
    }
    # TODO: ADD PARAMETERS TO ARGS FOR THESE 3 FUNCTIONS
    model = get_model(device=args.device)
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(model, optimizer, use_neve=args.use_neve)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.device == "cuda" and args.amp))

    # Init wandb project
    wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, config=args)

    # Training cycle
    for epoch in range(0, args.epochs):
        logs = train_epoch(model, data_loaders, optimizer, scheduler, scaler, args.device, args.amp, epoch=epoch)
        print(f"Epoch [{epoch + 1}]/[{args.epochs}]:\n{logs}\n")
        # Log on wandb project
        wandb.log(logs)

    # Save model...

    # End wandb run
    wandb.run.finish()


if __name__ == "__main__":
    main(get_args("classical"))
