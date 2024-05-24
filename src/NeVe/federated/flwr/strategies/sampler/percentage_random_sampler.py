import random

from flwr.common import FitRes, EvaluateRes
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy

from NeVe.federated.flwr.strategies.sampler import DefaultSampler
from NeVe.federated.flwr.strategies.sampler.logger import ClientSamplerLogger


class PercentageRandomSampler(DefaultSampler):
    def __init__(self, logger: ClientSamplerLogger, clients_sampling_percentage: float = 0.5):
        super().__init__(logger)
        self._clients_selection_percentage: float = clients_sampling_percentage

    def cleanup_results(self, results: list[tuple[ClientProxy, FitRes | EvaluateRes]], evaluate: bool = False) -> \
            list[tuple[ClientProxy, FitRes | EvaluateRes]]:
        return results

    def _sample_fit_clients(self, client_manager: ClientManager, sample_config_fz=None) -> list[ClientProxy]:
        clients = [client for _, client in client_manager.clients.items()]
        return random.sample(clients, int(self._clients_selection_percentage * len(clients)))
