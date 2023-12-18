import flwr as fl

from arguments import get_args
from my_flwr.strategies import weighted_average_fit, weighted_average_eval


def main(args):
    fl.server.start_server(server_address=args.server_address,
                           config=fl.server.ServerConfig(num_rounds=args.epochs),
                           strategy=fl.server.strategy.FedAvg(fit_metrics_aggregation_fn=weighted_average_fit,
                                                              evaluate_metrics_aggregation_fn=weighted_average_eval),
                           )


if __name__ == "__main__":
    main(get_args("server"))
