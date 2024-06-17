import torch


class NeVeHook:

    def __init__(self, name, module, momentum: float = 0.0) -> None:
        self._name = name
        self._module = module
        self._momentum = momentum

        # Avg velocity
        self._activations = []
        self._previous_activations = None
        self._rho = 0
        self._velocity = 0

        # Per neuron velocity
        self._neurons_activations = []
        self._previous_neurons_activations = None
        self._neurons_rho = 0
        self._neurons_velocity = 0

        self._active = True
        self._hook = module.register_forward_hook(self._hook_fn)

    def __del__(self):
        self._hook.remove()

    def _hook_fn(self, module, input, output):
        if not self._active:
            return
        current_activations = torch.movedim(output.detach(), 1, 0)
        current_neurons_activations = output.detach()
        current_activations = current_activations.reshape((current_activations.shape[0], -1))
        current_neurons_activations = current_neurons_activations.reshape((current_neurons_activations.shape[0], -1))
        self._activations.append(current_activations)
        self._neurons_activations.append(current_neurons_activations)

    def _update_rho(self):
        self._rho = torch.clamp(
            torch.nn.CosineSimilarity(dim=0)(
                self._get_current_activation().float(),
                self._previous_activations.float()
            ),
            -1., 1.
        )
        self._neurons_rho = torch.clamp(
            torch.nn.CosineSimilarity(dim=1)(
                self._get_current_neurons_activation().float(),
                self._previous_neurons_activations.float()
            ),
            -1., 1.
        )

    def _get_current_activation(self):
        return torch.cat(self._activations, dim=1).detach()

    def _get_current_neurons_activation(self):
        return torch.cat(self._neurons_activations, dim=1).detach()

    def get_velocity(self):
        self._update_rho()
        self._velocity = 1. - (self._rho + (self._momentum * self._velocity))
        self._neurons_velocity = abs(1. - (self._neurons_rho + (self._momentum * self._neurons_velocity)))
        return self._velocity.detach().cpu(), self._neurons_velocity.detach().cpu()

    def reset(self):
        self._previous_activations = self._get_current_activation()
        self._activations = []
        self._rho = 0
        self._previous_neurons_activations = self._get_current_neurons_activation()
        self._neurons_rho = 0
        self._neurons_activations = []

    def set_active(self, active: bool):
        self._active = active
