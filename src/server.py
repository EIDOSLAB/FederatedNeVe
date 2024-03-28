# ----- ----- ----- ----- -----
# TODO: FIX SRC IMPORTS IN A BETTER WAY
import sys
from pathlib import Path

import flwr as fl
import wandb
from flwr.server.strategy import FedAvg

FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# ----- ----- ----- ----- -----

from src.arguments import get_args
from src.utils import set_seeds
from src.my_flwr.strategies import weighted_average_fit, weighted_average_eval
from src.NeVe.federated.flwr.strategies.fedeneveavg import FedNeVeAvg


def main(args):
    # Init seeds
    set_seeds(args.seed)
    # Init wandb project
    wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, config=args)
    # Select strategy
    strategy = FedNeVeAvg if args.scheduler_name == "neve" else FedAvg
    # Start server
    fl.server.start_server(server_address=args.server_address,
                           config=fl.server.ServerConfig(num_rounds=args.epochs),
                           strategy=strategy(fit_metrics_aggregation_fn=weighted_average_fit,
                                             min_fit_clients=args.num_clients,
                                             min_evaluate_clients=args.num_clients,
                                             min_available_clients=args.num_clients,
                                             evaluate_metrics_aggregation_fn=weighted_average_eval)
                           )
    # End wandb run
    wandb.run.finish()


if __name__ == "__main__":
    main(get_args("server"))
