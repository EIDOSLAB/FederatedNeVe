import flwr as fl

from my_flwr.strategies import weighted_average_fit, weighted_average_eval


def main():
    fl.server.start_server(server_address="127.0.0.1:8080",
                           config=fl.server.ServerConfig(num_rounds=3),
                           strategy=fl.server.strategy.FedAvg(fit_metrics_aggregation_fn=weighted_average_fit,
                                                              evaluate_metrics_aggregation_fn=weighted_average_eval),
                           )


if __name__ == "__main__":
    main()
