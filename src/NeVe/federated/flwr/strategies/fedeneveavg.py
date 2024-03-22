from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FedNeVeAvg(FedAvg):
    """Federated NeVe Averaging strategy.

    Implementation based on Federated Averaging strategy https://arxiv.org/abs/1602.05629
    and updated using our NeVe interface.
    """

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
