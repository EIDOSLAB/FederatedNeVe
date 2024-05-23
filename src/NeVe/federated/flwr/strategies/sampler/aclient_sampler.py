import abc
from abc import ABC

from flwr.common import FitRes, EvaluateRes
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy

from NeVe.federated.flwr.strategies.sampler.logger import ClientSamplerLogger


class ClientSampler(ABC):
    def __init__(self, logger: ClientSamplerLogger):
        self._clients_mapping: dict[str, int] = {}
        self._logger: ClientSamplerLogger = logger

    def sample_fit_clients(self, client_manager: ClientManager, sample_config_fz=None) -> list[ClientProxy]:
        sampled_clients = self._sample_fit_clients(client_manager, sample_config_fz)
        if self._logger:
            self._logger.log_sampled_fit_clients(sampled_clients, self._clients_mapping)
        return sampled_clients

    @abc.abstractmethod
    def update_clients_mapping(self, client_manager: ClientManager | None,
                               new_clients_mapping: list[tuple[ClientProxy, int]]):
        pass

    @abc.abstractmethod
    def update_clients_data(self, new_results: list[tuple[ClientProxy, FitRes]]):
        pass

    @abc.abstractmethod
    def cleanup_results(self, results: list[tuple[ClientProxy, FitRes | EvaluateRes]], evaluate: bool = False) -> \
            list[tuple[ClientProxy, FitRes | EvaluateRes]]:
        pass

    @abc.abstractmethod
    def _sample_fit_clients(self, client_manager: ClientManager, sample_config_fz=None) -> list[ClientProxy]:
        pass
