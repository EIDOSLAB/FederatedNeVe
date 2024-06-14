from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy

from NeVe.federated.flwr.strategies.sampler import PercentageRandomSampler
from NeVe.federated.flwr.strategies.sampler.logger import ClientSamplerLogger


class PercentageRGroupsSampler(PercentageRandomSampler):
    def __init__(self, logger: ClientSamplerLogger,
                 clients_sampling_percentage: float = 0.5, sampling_wait_epochs: int = 10):
        super().__init__(logger, clients_sampling_percentage=clients_sampling_percentage,
                         sampling_wait_epochs=sampling_wait_epochs)
        self._clients_selection_rr_current_idx: int = 0

    def _sample_fit_clients(self, client_manager: ClientManager, epoch: int,
                            sample_config_fz=None) -> list[ClientProxy]:
        if epoch < self._sampling_wait_epochs:
            return self._default_fit_sample(client_manager, epoch, sample_config_fz)

        # Get sampling configuration
        group_sample_size, min_num_clients = sample_config_fz(
            client_manager.num_available()
        )

        # Get all clients and update the sample_size of the groups
        clients = [client for _, client in client_manager.clients.items()]
        if group_sample_size > int(self._clients_selection_percentage * len(clients)):
            group_sample_size = int(self._clients_selection_percentage * len(clients))
        if group_sample_size < 1:
            group_sample_size = 1

        # Get the first element of the group
        current_idx = (self._clients_selection_rr_current_idx * group_sample_size) % len(clients)
        # To avoid the round-robin group index to become too big we reset it to 0 when the % gives 0
        if current_idx == 0:
            self._clients_selection_rr_current_idx = 0

        # Sample all clients between the current index and the current index + the group size
        clients = clients[current_idx:current_idx + group_sample_size]

        # Update the current round-robin group index
        self._clients_selection_rr_current_idx += 1
        return clients
