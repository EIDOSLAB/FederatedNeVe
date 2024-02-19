# ----- ----- ----- ----- -----
# TODO: FIX SRC IMPORTS IN A BETTER WAY
import sys
from pathlib import Path

import flwr as fl

import wandb

FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# ----- ----- ----- ----- -----

from src.arguments import get_args
from src.my_flwr.strategies import weighted_average_fit, weighted_average_eval
from src.utils import set_seeds


def main(args):
    # Init seeds
    set_seeds(args.seed)
    # Init wandb project
    wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, config=args)
    # Start server
    fl.server.start_server(server_address=args.server_address,
                           config=fl.server.ServerConfig(num_rounds=args.epochs),
                           strategy=fl.server.strategy.FedAvg(fit_metrics_aggregation_fn=weighted_average_fit,
                                                              evaluate_metrics_aggregation_fn=weighted_average_eval)
                           )
    # End wandb run
    wandb.run.finish()


if __name__ == "__main__":
    main(get_args("server"))
