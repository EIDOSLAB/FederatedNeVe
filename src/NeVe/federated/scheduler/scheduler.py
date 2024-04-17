import os.path

import torch
from torch import nn

from NeVe.scheduler import NeVeScheduler


class FederatedNeVeScheduler(NeVeScheduler):
    def __init__(self, model: nn.Module, lr_scheduler, velocity_momentum: float = 0.5, stop_threshold: float = 0.001,
                 save_path: str = "../fclients_data/", client_id: int = 0, only_last_layer: bool = False):
        super().__init__(model, lr_scheduler, velocity_momentum=velocity_momentum, stop_threshold=stop_threshold,
                         only_last_layer=only_last_layer)
        self.base_save_path = save_path
        self._activations_save_path = os.path.join(save_path, "activations")
        self._client_id = client_id
        # Make sure the folder exists
        os.makedirs(self._activations_save_path, exist_ok=True)

    def save_activations(self):
        assert self._activations_save_path
        # For each hook we save the previous_activations
        for h in self._hooks:
            activations = torch.empty(
                (len(self._hooks[h]._previous_activations), self._hooks[h]._previous_activations[0].shape[0])
            )
            for index, activation in enumerate(self._hooks[h]._previous_activations):
                activations[index] = activation
            torch.save(activations, os.path.join(self._activations_save_path, str(self._client_id) + "_" + h + ".pt"))

    def load_activations(self, device):
        assert self._activations_save_path and os.path.exists(self._activations_save_path)
        # For each hook we load the previous_activations
        for h in self._hooks:
            path = os.path.join(self._activations_save_path, str(self._client_id) + "_" + h + ".pt")
            if os.path.exists(path):
                self._hooks[h]._previous_activations = torch.load(path).to(device)

    def load_state_dicts(self, lr_state_dict, velocity_state_dict):
        self._lr_scheduler.load_state_dict(lr_state_dict)
        self._velocity_cache = velocity_state_dict

    def state_dicts(self):
        return self._lr_scheduler.state_dict(), self._velocity_cache
