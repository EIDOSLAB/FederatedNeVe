import flwr as fl

from src.arguments import get_args
from src.my_flwr.strategies import weighted_average_fit, weighted_average_eval
from src.utils import set_seeds


def main(args):
    # Init seeds
    set_seeds(args.seed)
    fl.server.start_server(server_address=args.server_address,
                           config=fl.server.ServerConfig(num_rounds=args.epochs),
                           strategy=fl.server.strategy.FedAvg(fit_metrics_aggregation_fn=weighted_average_fit,
                                                              evaluate_metrics_aggregation_fn=weighted_average_eval)
                           )


if __name__ == "__main__":
    main(get_args("server"))
