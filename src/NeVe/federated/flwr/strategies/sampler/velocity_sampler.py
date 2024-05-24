from flwr.common import FitRes, EvaluateRes
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy

from NeVe.federated.flwr.strategies.sampler import PercentageRandomSampler
from NeVe.federated.flwr.strategies.sampler.logger import ClientSamplerLogger


class VelocitySampler(PercentageRandomSampler):
    def __init__(self, logger: ClientSamplerLogger, clients_sampling_percentage: float = 0.5):
        super().__init__(logger, clients_sampling_percentage=clients_sampling_percentage)
        self.clients_velocity: dict[int, float] = {}

    def update_clients_data(self, new_results: list[tuple[ClientProxy, FitRes]]):
        # Update current velocity values
        for client_proxy, results in new_results:
            idx = self._clients_mapping[client_proxy.cid]
            self.clients_velocity[idx] = results.metrics.get("neve.model_avg_value", 0.0)

    def cleanup_results(self, results: list[tuple[ClientProxy, FitRes | EvaluateRes]], evaluate: bool = False) -> \
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

    def _sample_fit_clients(self, client_manager: ClientManager, sample_config_fz=None) -> list[ClientProxy]:
        # Get list of all available clients
        clients = [client for _, client in client_manager.clients.items()]

        # Select clients with no velocity
        clients_no_velocity = [client for client in clients if client.cid in self._clients_mapping.keys() and
                               self._clients_mapping[client.cid] not in self.clients_velocity.keys()]

        # Select remaining clients with the highest velocity until we reached the desired number of sampled clients
        required_clients_2_sample = int(self._clients_selection_percentage * len(clients))
        clients_highest_velocity = []
        clients_ordered_by_velocity = sorted(self.clients_velocity.items(), key=lambda item: item[1], reverse=True)
        # Cycle all the clients, starting from the ones with the highest velocity
        for client_ordered_idx, _ in clients_ordered_by_velocity:
            # If we have enough clients, skip
            if len(clients_no_velocity) + len(clients_highest_velocity) >= required_clients_2_sample:
                break
            # Get the client reference from the list of clients
            for client in clients:
                if self._clients_mapping[client.cid] == client_ordered_idx:
                    clients_highest_velocity.append(client)
                    break

        # The sampled clients are the merge of the 2 lists
        return clients_no_velocity + clients_highest_velocity
