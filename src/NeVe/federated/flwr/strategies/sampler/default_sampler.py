from flwr.common import FitRes, EvaluateRes
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy

from NeVe.federated.flwr.strategies.sampler import ClientSampler
from NeVe.federated.flwr.strategies.sampler.logger import ClientSamplerLogger


class DefaultSampler(ClientSampler):
    def __init__(self, logger: ClientSamplerLogger):
        super().__init__(logger)
        self._clients_mapping_current_id: int = 0

    def _update_clients_mapping(self, client_manager: ClientManager):
        for cid, _ in client_manager.clients.items():
            if cid not in self._clients_mapping.keys():
                self._clients_mapping[cid] = self._clients_mapping_current_id
                self._clients_mapping_current_id += 1

    def update_clients_data(self, new_results: list[tuple[ClientProxy, FitRes]]):
        return

    def cleanup_results(self, results: list[tuple[ClientProxy, FitRes | EvaluateRes]], evaluate: bool = False) -> \
            list[tuple[ClientProxy, FitRes | EvaluateRes]]:
        return results

    def _sample_fit_clients(self, client_manager: ClientManager, sample_config_fz=None) -> list[ClientProxy]:
        # Sample clients-size
        sample_size, min_num_clients = sample_config_fz(
            client_manager.num_available()
        )
        return client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
