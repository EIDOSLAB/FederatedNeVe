# ----- ----- ----- ----- -----
# TODO: FIX SRC IMPORTS IN A BETTER WAY
import sys
from pathlib import Path

import torch
import wandb

FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# ----- ----- ----- ----- -----

from src.arguments import get_args
from src.dataloaders import get_dataset, prepare_data
from src.models import get_model
from src.utils import set_seeds, get_optimizer, get_scheduler
from src.utils.trainer import train_epoch, run
from src.NeVe import NeVeOptimizer


def main(args):
    # Init seeds
    set_seeds(args.seed)

    # TODO: ADD PARAMETERS TO ARGS FOR THESE 3 FUNCTIONS
    model = get_model(dataset=args.dataset_name, device=args.device)
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(model, optimizer, use_neve=args.use_neve)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.device == "cuda" and args.amp))

    # Load Data
    train, test, aux = get_dataset(args.dataset_root, args.dataset_name, seed=args.seed, generate_aux_set=args.use_neve)
    train_loaders, val_loaders, test_loader, aux_loader = prepare_data(train, test, aux, num_clients=1,
                                                                       seed=args.seed, batch_size=args.batch_size)
    train_loader, val_loader = train_loaders[0], val_loaders[0]
    data_loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "aux": aux_loader,
    }
    # Init seeds
    set_seeds(args.seed)  # Just to be sure to be in the same spot whatever we generated the aux dataset or not

    # Init wandb project
    wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, config=args)

    # NeVe init
    if args.use_neve and "aux" in data_loaders.keys() and data_loaders["aux"] and isinstance(scheduler, NeVeOptimizer):
        with scheduler:
            _ = run(model, data_loaders["aux"], None, scaler, args.device, args.amp, -1, "Aux")
        _ = scheduler.step(init_step=True)

    # Training cycle
    for epoch in range(0, args.epochs):
        logs, neve_data = train_epoch(model, data_loaders, optimizer=optimizer, scheduler=scheduler,
                                      grad_scaler=scaler, device=args.device, amp=args.amp, epoch=epoch)
        print("\n-----")
        print(f"Epoch [{epoch + 1}]/[{args.epochs}]:")
        print(f"LR: {logs['lr']}")
        print(f"Train: {logs['train']}")
        print(f"Val: {logs['val']}")
        print(f"Test: {logs['test']}")
        if args.use_neve and 'aux' in logs.keys():
            print(f"Aux: avg.vel. {logs['aux']['neve']['model_avg_value']} ")
        print("-----\n")
        if neve_data and not neve_data.continue_training:
            # TODO: ADD BREAK CONDITION
            print("Training stopped since neve velocity dropped below the threshold.")
        # Log on wandb project
        wandb.log(logs)

    # Save model...

    # End wandb run
    wandb.run.finish()


if __name__ == "__main__":
    main(get_args("classical"))
