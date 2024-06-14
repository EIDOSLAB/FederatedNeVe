# ----- ----- ----- ----- -----
# TODO: FIX SRC IMPORTS IN A BETTER WAY
import sys
from pathlib import Path

import flwr as fl
import wandb
from flwr.server.strategy import FedAvg

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
# ----- ----- ----- ----- -----

from src.arguments import get_args
from src.utils import set_seeds
from src.my_flwr.strategies import weighted_average_fit, weighted_average_eval
from src.NeVe.federated.flwr.strategies.fedeneveavg import FedNeVeAvg
from src.NeVe.federated.flwr.strategies.sampler import get_client_sampler


def main(args):
    # TODO: PLOTTARE LA DISTRIBUZIONE DEI DATI
    # TODO: RITORNARE LA DISTRIBUZIONE DEI DATI PER DATI IID
    # TODO: MATRICE DI CONFUSIONE DELLE CLASSI PER CAPIRE DOVE HO PROBLEMI, QUESTA DEVE ESSERCI PER OGNI EPOCA
    # TODO: AGGIORNARE LO SCHEDULER AD OGNI EPOCA DA OGNI CLIENT (ES. NEL VALIDATE CHE è FATTO SEMPRE DA TUTTI)

    # Init seeds
    set_seeds(args.seed)
    # Init wandb project
    wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, config=args, tags=args.wandb_tags)
    # Select strategy
    strategy_type = FedNeVeAvg if args.neve_active else FedAvg
    if args.neve_active:
        strategy = strategy_type(
            client_sampler=get_client_sampler(args.clients_sampling_method,
                                              sampling_percentage=args.clients_sampling_percentage,
                                              sampling_wait_epochs=args.clients_sampling_wait_epochs,
                                              sampling_velocity_aging=args.clients_sampling_velocity_aging,
                                              sampling_highest_velocity=args.clients_sampling_highest_velocity),
            fit_metrics_aggregation_fn=weighted_average_fit,
            min_fit_clients=args.min_fit_clients,
            min_evaluate_clients=args.min_evaluate_clients,
            min_available_clients=args.num_clients,
            evaluate_metrics_aggregation_fn=weighted_average_eval
        )
    else:
        strategy = strategy_type(
            fit_metrics_aggregation_fn=weighted_average_fit,
            min_fit_clients=args.min_fit_clients,
            min_evaluate_clients=args.min_evaluate_clients,
            min_available_clients=args.num_clients,
            evaluate_metrics_aggregation_fn=weighted_average_eval
        )

    # Start server
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.epochs),
        strategy=strategy
    )
    # End wandb run
    wandb.run.finish()


if __name__ == "__main__":
    main(get_args("server"))
