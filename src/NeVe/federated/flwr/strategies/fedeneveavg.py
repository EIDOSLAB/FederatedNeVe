import random
from typing import Callable, Dict, List, Optional, Tuple

import wandb
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
            clients_selection_method: str = "default",
            clients_selection_percentage: float = 0.5
    ) -> None:
        super().__init__(fraction_fit=fraction_fit, fraction_evaluate=fraction_evaluate,
                         min_fit_clients=min_fit_clients, min_evaluate_clients=min_evaluate_clients,
                         min_available_clients=min_available_clients, evaluate_fn=evaluate_fn,
                         on_fit_config_fn=on_fit_config_fn, on_evaluate_config_fn=on_evaluate_config_fn,
                         accept_failures=accept_failures, initial_parameters=initial_parameters,
                         fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                         evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn, inplace=inplace)
        self.clients_velocity: dict[int, float] = {}
        self.even_fit: bool = False
        self.clients_selection_method: str = clients_selection_method
        self.clients_selection_percentage: float = clients_selection_percentage
        self.clients_selection_rr_current_idx: int = 0
        # Make sure the percentage is normalized between 0 and 1
        if self.clients_selection_percentage > 1.0:
            self.clients_selection_percentage /= 100

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
        results = self._cleanup_results(results)

        # Update current velocity values
        for client_proxy, result in results:
            cid = result.metrics.get("client_id", -1)
            self.clients_velocity[cid] = result.metrics.get("neve.model_avg_value", 0.0)

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
        clients = self._sample_clients(client_manager)

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

        results = self._cleanup_results(results, evaluate=True)

        return super().aggregate_evaluate(server_round, results, failures)

    @staticmethod
    def neve_results_cleanup(results: list[tuple[ClientProxy, FitRes | EvaluateRes]], evaluate: bool = False) -> \
            list[tuple[ClientProxy, FitRes | EvaluateRes]]:
        if evaluate:
            return results
        cleaned_results = []
        for client_data, res in results:
            # Do not consider clients if their returned data is None (velocity under threshold)
            if not res.parameters.tensors:
                continue
            cleaned_results.append((client_data, res))
        return cleaned_results

    def _cleanup_results(self, results: list[tuple[ClientProxy, FitRes | EvaluateRes]], evaluate: bool = False) -> \
            list[tuple[ClientProxy, FitRes | EvaluateRes]]:
        match self.clients_selection_method:
            case "default":
                return results
            case "default_percentage":
                return results
            case "default_percentage_random":
                return results
            case "velocity":
                return FedNeVeAvg.neve_results_cleanup(results, evaluate=evaluate)
            case _:
                return results

    def _sample_clients(self, client_manager: ClientManager):
        # Sample clients-size
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        # We sample the clients based on the selection method
        match self.clients_selection_method:
            # All clients selected
            case "default":
                clients = client_manager.sample(
                    num_clients=sample_size, min_num_clients=min_num_clients
                )
            # N% of clients selected, and then selected at round-robin
            case "default_percentage":
                clients = [client for _, client in client_manager.clients.items()]
                if sample_size > int(self.clients_selection_percentage * len(clients)):
                    sample_size = int(self.clients_selection_percentage * len(clients))
                if sample_size < 0:
                    sample_size = 1
                current_idx = (self.clients_selection_rr_current_idx * sample_size) % len(clients)
                clients = clients[current_idx:current_idx + sample_size]
                self.clients_selection_rr_current_idx += 1
            # N% of clients selected randomly
            case "default_percentage_random":
                clients = [client for _, client in client_manager.clients.items()]
                clients = random.sample(clients, int(self.clients_selection_percentage * len(clients)))
            # N% clients selected with the highest velocity
            case "velocity":
                # Prendo i clients
                clients = [(idx, client) for idx, client in client_manager.clients.items()]
                # Prendo i clients che non hanno velocity associate
                clients_no_velocity = [client for idx, client in clients if
                                       int(idx) not in self.clients_velocity.keys()]
                # Dei rimanenti ritorno i clients con la velocity più alta fino ad arrivare al numero minimo richiesto
                clients_highest_velocity = []
                required_clients = int(self.clients_selection_percentage * len(clients))
                ordered_clients = sorted(self.clients_velocity.items(), key=lambda item: item[1], reverse=True)
                for _, client in ordered_clients:
                    if len(clients_no_velocity) + len(clients_highest_velocity) >= required_clients:
                        break
                    clients_highest_velocity.append(client)
                # I clients da usare sono la somma delle due liste
                clients = clients_no_velocity + clients_highest_velocity
            # By default, we just select them all
            case _:
                clients = client_manager.sample(
                    num_clients=sample_size, min_num_clients=min_num_clients
                )
        # Log which clients have been chosen
        selected_clients = [0 for _ in client_manager.clients.items()]
        for idx, client in client_manager.clients.items():
            for client_s in clients:
                if client == client_s:
                    selected_clients[idx] = 1
                    break
        selected_clients_logs = {
            "selected_clients": wandb.Table(data=[[v] for v in selected_clients], columns=["value"])
        }
        wandb.log(selected_clients_logs, commit=False)
        return clients
