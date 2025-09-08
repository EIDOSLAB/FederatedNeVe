from functools import reduce
from logging import WARNING
from typing import Dict, List, Optional, Tuple, Union, Callable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays, NDArrays, MetricsAggregationFn,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate

from my_federated.my_flwr.strategies.FedAvgWConfig import FedAvgWConfig
from my_federated.my_flwr.strategies.strategy_data import StrategyData


class FedNeVeAvg(FedAvgWConfig):
    def __init__(
            self,
            *,
            inverse_velocity: bool = False,
            temperature: float = 1.0,
            epochs: int = 100,
            velocity_aggregation_fn: str = None,
            strategy_data: StrategyData = None,
            fraction_fit: float = 1.0,
            fraction_evaluate: float = 1.0,
            min_fit_clients: int = 2,
            min_evaluate_clients: int = 2,
            min_available_clients: int = 2,
            evaluate_fn: Optional[
                Callable[
                    [int, NDArrays, Dict[str, Scalar]],
                    Optional[Tuple[float, Dict[str, Scalar]]],
                ]
            ] = None,
            on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
            accept_failures: bool = True,
            initial_parameters: Optional[Parameters] = None,
            fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
            inplace: bool = True,
    ) -> None:
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate,
                         min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients,
                         min_available_clients=min_available_clients, evaluate_fn=evaluate_fn,
                         strategy_data=strategy_data,
                         on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn,
                         accept_failures=accept_failures, initial_parameters=initial_parameters,
                         fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                         evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace, )
        self.inverse_velocity: bool = inverse_velocity
        self.temperature: float = temperature
        self.velocity_aggregation_fn: str = velocity_aggregation_fn
        self._min_fit_clients = min_fit_clients
        self.data_matrix = np.zeros((self._min_fit_clients, epochs))
        self.current_epoch = 0
        self.max_epochs = epochs

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using velocity-based weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.inplace:
            # Does in-place velocity-based weighted average of results
            aggregated_ndarrays = self.aggregate_inplace(results)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.metrics.get("neve.velocity", 0.0))
                for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        self.update_model_parameters(parameters_aggregated)
        self.test_aggregated_model(server_round)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        self.current_epoch += 1
        return parameters_aggregated, metrics_aggregated

    def aggregate_inplace(self, results: List[Tuple[ClientProxy, FitRes]]) -> NDArrays:
        """Compute in-place velocity-based weighted average."""
        match self.velocity_aggregation_fn:
            case "avg":
                scaling_factors = aggregate_avg(results, self.inverse_velocity)
            case "soft_exp":
                scaling_factors = aggregate_soft_exp(results, self.inverse_velocity, self.temperature)
            case _:
                scaling_factors = aggregate_avg(results, self.inverse_velocity)
        if len(scaling_factors) != self._min_fit_clients:
            print(f"fed_neve_avg.py -> aggregate_inplace: length of scaling_factors ({len(scaling_factors)}) "
                  f"is different than min_fit_clients ({self.min_fit_clients}). Zeros will be used.")
            # Usa un vettore di zeri
            self.data_matrix[:, self.current_epoch] = np.zeros(self.min_fit_clients)
        else:
            self.data_matrix[:, self.current_epoch] = np.array(scaling_factors)

        data_to_log = {
            "FedNeVeAvg": {
                "scaling_factors": {key: value for key, value in enumerate(scaling_factors)},
            }
        }
        # At the last epoch, also print a heatmap that shows all the accumulated data
        if self.current_epoch >= self.max_epochs - 1:
            plt.figure(figsize=(12, 6))
            sns.heatmap(self.data_matrix, cmap="viridis", cbar=True)
            plt.xlabel("Epoch")
            plt.ylabel("Client")
            data_to_log["FedNeVeAvg"]["scaling_factors_figure"] = wandb.Image(plt)
            plt.close()
        wandb.log(data_to_log, commit=False)

        # Let's do in-place aggregation
        # Get first result, then add up each other
        params = [
            scaling_factors[0] * x for x in parameters_to_ndarrays(results[0][1].parameters)
        ]
        for i, (_, fit_res) in enumerate(results[1:]):
            res = (
                scaling_factors[i + 1] * x
                for x in parameters_to_ndarrays(fit_res.parameters)
            )
            params = [reduce(np.add, layer_updates) for layer_updates in zip(params, res)]

        return params


def aggregate_avg(results: List[Tuple[ClientProxy, FitRes]], inverse_velocity: bool = False, eps: float = 1e-6):
    # Count total velocity
    weights = [fit_res.metrics.get("neve.velocity", 0.0) for (_, fit_res) in results]
    velocity_total = sum(weights)

    if inverse_velocity:
        weights = [velocity_total - weight for weight in weights]

    # Compute scaling factors for each result normalized between [0, 1]
    velocity_total = sum(weights) + eps
    scaling_factors = [weight / velocity_total for weight in weights]
    return scaling_factors


def aggregate_soft_exp(results: List[Tuple[ClientProxy, FitRes]], inverse_velocity: bool = False,
                       temperature: float = 1.0):
    assert temperature != 0, f"aggregate_soft_exp -> temperature must be different from 0"
    weights = [fit_res.metrics.get("neve.velocity", 0.0) / temperature for (_, fit_res) in results]
    soft_exp_fn = torch.nn.Softmin if inverse_velocity else torch.nn.Softmax
    soft_exp_fn = soft_exp_fn(dim=0)
    scaling_factors = soft_exp_fn(torch.tensor(weights)).tolist()
    return scaling_factors
