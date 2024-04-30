from typing import Callable, Dict, List, Optional, Tuple

from flwr.common import (
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
)
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FedNeVeAvg(FedAvg):
    """Federated NeVe Averaging strategy.

    Implementation based on Federated Averaging strategy https://arxiv.org/abs/1602.05629
    and updated using our NeVe interface.

    Parameters
    ----------
    fraction_fit : float, optional
        Fraction of clients used during training. In case `min_fit_clients`
        is larger than `fraction_fit * available_clients`, `min_fit_clients`
        will still be sampled. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. In case `min_evaluate_clients`
        is larger than `fraction_evaluate * available_clients`,
        `min_evaluate_clients` will still be sampled. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]],Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    inplace : bool (default: True)
        Enable (True) or disable (False) in-place aggregation of model updates.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
            self,
            *,
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
                         on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn,
                         accept_failures=accept_failures, initial_parameters=initial_parameters,
                         fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                         evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)
        self.clients_velocity: list[tuple[ClientProxy, float]] = []

    def aggregate_fit(
            self,
            server_round: int,
            results: list[tuple[ClientProxy, FitRes]],
            failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Cleanup results where there is no parameters because velocity is too low
        results = self.neve_results_check(results)

        return super().aggregate_fit(server_round, results, failures)

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        # TODO: QUA USO LA METà DEI CLIENTS
        if sample_size > min_num_clients / 2:
            sample_size = int(min_num_clients / 2)

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_evaluate(
            self,
            server_round: int,
            results: list[tuple[ClientProxy, EvaluateRes]],
            failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # TODO: AGGIUNGERE VELOCITY A SELF.CLIENTS_VELOCITY

        # Cleanup results where there is no parameters because velocity is too low
        results = self.neve_results_check(results, evaluate=True)

        return super().aggregate_evaluate(server_round, results, failures)

    @staticmethod
    def neve_results_check(results: list[tuple[ClientProxy, FitRes | EvaluateRes]], evaluate: bool = False) -> \
            list[tuple[ClientProxy, FitRes | EvaluateRes]]:
        cleaned_results = []
        if not evaluate:
            for client_data, res in results:
                # Do not consider the client if the returned data is None (velocity under threshold)
                if not res.parameters.tensors:
                    continue
                cleaned_results.append((client_data, res))
        else:
            cleaned_results = results
        return cleaned_results
