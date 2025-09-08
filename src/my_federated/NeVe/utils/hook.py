import torch


class NeVeHook:

    def __init__(self, name, module, momentum: float = 0.0, data_distribution: list[float] | None = None) -> None:
        self._name = name
        self._module = module
        self._momentum = momentum
        self._data_distribution = torch.tensor(data_distribution) if data_distribution else None

        # Avg velocity
        self._activations = []
        self._previous_activations = None
        self._rho = 0
        self._velocity = 0

        self._active = True
        self._hook = module.register_forward_hook(self._hook_fn)

    def update_data_distribution(self, new_data_distribution):
        self._data_distribution = torch.tensor(new_data_distribution) if new_data_distribution else None

    def __del__(self):
        self._hook.remove()

    def _hook_fn(self, module, input, output):
        if not self._active:
            return
        current_activations = torch.movedim(output.detach(), 1, 0)
        current_activations = current_activations.reshape((current_activations.shape[0], -1))
        self._activations.append(current_activations)

    def _update_rho(self):
        self._rho = torch.clamp(
            torch.nn.CosineSimilarity(dim=1)(
                self._get_current_activation().float(),
                self._previous_activations.float()
            ),
            -1., 1.
        )

    def _get_current_activation(self):
        return torch.cat(self._activations, dim=1).detach()

    def get_velocity(self):
        self._update_rho()
        self._velocity = torch.sub(1.0, (self._rho + (self._momentum * self._velocity)))

        if self._data_distribution is not None:
            if self._velocity.device != self._data_distribution.device:
                self._data_distribution = self._data_distribution.to(self._velocity.device)
            if self._velocity.shape == self._data_distribution.shape:
                self._velocity *= self._data_distribution
            else:
                print(f"NeVeHook - Velocity was not normalized by the data_distribution values since their shapes "
                      f"are different: velocity shape: {str(self._velocity.shape)}, "
                      f"data_distribution shape: {str(self._data_distribution.shape)}")
        return self._velocity.detach().cpu()

    def reset(self):
        self._previous_activations = self._get_current_activation()
        self._activations = []
        self._rho = 0

    def full_reset(self):
        self._activations = []
        self._previous_activations = None
        self._rho = 0
        self._velocity = 0

    def set_active(self, active: bool):
        self._active = active
