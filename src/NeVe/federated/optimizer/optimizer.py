import os.path

import torch
from torch import nn

from NeVe.optimizer import NeVeOptimizer


class FederatedNeVeOptimizer(NeVeOptimizer):
    def __init__(self, model: nn.Module, scheduler, velocity_momentum: float = 0.5, stop_threshold: float = 0.001,
                 activations_save_path: str = "../activations/", client_id: int = 0):
        super().__init__(model, scheduler, velocity_momentum=velocity_momentum, stop_threshold=stop_threshold)
        self._activations_save_path = activations_save_path
        self._client_id = client_id

    def save_activations(self):
        assert self._activations_save_path
        os.makedirs(self._activations_save_path, exist_ok=True)
        for h in self._hooks:
            activations = torch.empty(
                (len(self._hooks[h]._previous_activations), self._hooks[h]._previous_activations[0].shape[0]))
            for index, activation in enumerate(self._hooks[h]._previous_activations):
                activations[index] = activation
            torch.save(activations, os.path.join(self._activations_save_path, str(self._client_id) + "_" + h + ".pt"))

    def load_activations(self, device):
        assert self._activations_save_path and os.path.exists(self._activations_save_path)
        for h in self._hooks:
            path = os.path.join(self._activations_save_path, str(self._client_id) + "_" + h + ".pt")
            if os.path.exists(path):
                activations = torch.load(path).to(device)
                self._hooks[h]._previous_activations = activations
