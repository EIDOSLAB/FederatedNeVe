from flwr.common import FitRes, EvaluateRes
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy

from NeVe.federated.flwr.strategies.sampler import PercentageRandomSampler
from NeVe.federated.flwr.strategies.sampler.logger import ClientSamplerLogger


class VelocitySampler(PercentageRandomSampler):
    def __init__(self, logger: ClientSamplerLogger,
                 clients_sampling_percentage: float = 0.5, sampling_wait_epochs: int = 10,
                 sampling_velocity_aging: float = 0.01, sampling_highest_velocity: bool = True,
                 sampling_min_epochs: int = 2):
        super().__init__(logger, clients_sampling_percentage=clients_sampling_percentage,
                         sampling_wait_epochs=sampling_wait_epochs)
        self.clients_velocity: dict[int, float] = {}
        self.sampling_velocity_aging: float = sampling_velocity_aging
        self.sampling_highest_velocity: bool = sampling_highest_velocity
        # If we select the lowest velocity we also need to reduce the velocity when aging is performed
        if not self.sampling_highest_velocity:
            self.sampling_velocity_aging *= -1
        self._previous_selected_clients = []
        # TODO: MANAGE THIS PARAMETER, ADD IT TO ARGUMENTS.PY
        self.sampling_min_epochs = sampling_min_epochs
        if self.sampling_min_epochs < 1:
            self.sampling_min_epochs = 1
        self._sampling_min_epochs_count = self.sampling_min_epochs

    def update_clients_data(self, new_results: list[tuple[ClientProxy, FitRes]]):
        # Apply aging to all velocities BEFORE updating velocities of sampled clients
        # This way only not-sampled clients will have their velocity decayed
        for idx in self.clients_velocity.keys():
            self.clients_velocity[idx] *= (1 + self.sampling_velocity_aging)

        # Update velocity values of sampled clients
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

    def _sample_fit_clients(self, client_manager: ClientManager, epoch: int,
                            sample_config_fz=None) -> list[ClientProxy]:
        if epoch < self._sampling_wait_epochs:
            return self._default_fit_sample(client_manager, epoch, sample_config_fz)

        # Return previous selected clients if we are still in the range of epochs in which we select the same clients
        self._sampling_min_epochs_count -= 1
        if self._previous_selected_clients and self._sampling_min_epochs_count > 0:
            return self._previous_selected_clients

        # Get list of all available clients
        clients = [client for _, client in client_manager.clients.items()]

        # Select clients with no velocity
        clients_no_velocity = [client for client in clients if client.cid in self._clients_mapping.keys() and
                               self._clients_mapping[client.cid] not in self.clients_velocity.keys()]

        # Select remaining clients with the highest velocity until we reached the desired number of sampled clients
        required_clients_2_sample = int(self._clients_selection_percentage * len(clients))
        clients_best_velocity = []
        clients_ordered_by_velocity = sorted(self.clients_velocity.items(), key=lambda item: item[1],
                                             reverse=self.sampling_highest_velocity)
        # Cycle all the clients, starting from the ones with the highest velocity
        for client_ordered_idx, _ in clients_ordered_by_velocity:
            # If we have enough clients, skip
            if len(clients_no_velocity) + len(clients_best_velocity) >= required_clients_2_sample:
                break
            # Get the client reference from the list of clients
            for client in clients:
                if self._clients_mapping[client.cid] == client_ordered_idx:
                    clients_best_velocity.append(client)
                    break

        # The sampled clients are the merge of the 2 lists
        selected_clients = clients_no_velocity + clients_best_velocity
        # Update the list containing the previous selected clients and init the counter for their sampling
        self._previous_selected_clients = selected_clients
        self._sampling_min_epochs_count = self.sampling_min_epochs
        return selected_clients
