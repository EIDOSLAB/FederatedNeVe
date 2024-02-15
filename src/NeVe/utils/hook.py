import copy

import torch


class NeVeHook:

    def __init__(self, name, module, momentum: float = 0.0) -> None:
        self._name = name
        self._module = module
        self._momentum = momentum

        self._activations = []
        self._previous_activations = None
        self._rho = 0
        self._velocity = 0

        self._active = True
        self._hook = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        if not self._active:
            return
        output_tensor = output.detach()
        current_activations = torch.movedim(output_tensor, 1, 0)
        current_activations = current_activations.reshape((current_activations.shape[0], -1))
        self._activations.append(current_activations)

    def _update_rho(self):
        similarity = torch.nn.CosineSimilarity(dim=0)(
            self._get_current_activation().float(),
            self._previous_activations.float()
        )
        self._rho = torch.clamp(similarity, -1., 1.)

    def _get_current_activation(self):
        return torch.cat(self._activations, dim=1)

    def get_velocity(self):
        self._update_rho()
        self._velocity = 1. - (self._rho + (self._momentum * self._velocity))
        return self._velocity

    def reset(self):
        self._previous_activations = copy.deepcopy(self._get_current_activation().detach())
        self._activations = []
        self._rho = 0

    def close(self):
        self._hook.remove()

    def set_active(self, active: bool):
        self._active = active
