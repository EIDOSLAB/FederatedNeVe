import torch
from torch import nn

from NEq.utils import NEqHook


@torch.no_grad()
def _find_module_by_name(model, name):
    module = model
    splitted_name = name.split(".")
    for idx, sub in enumerate(splitted_name):
        if idx < len(splitted_name):
            module = getattr(module, sub)

    return module


def _log_masks(model, grad_mask, total_neurons):
    frozen_neurons = 0

    per_layer_frozen_neurons = {}

    for k in grad_mask:
        frozen_neurons += grad_mask[k].shape[0]

        module = _find_module_by_name(model, k)

        # Log the percentage of frozen neurons per layer
        per_layer_frozen_neurons[f"{k}"] = grad_mask[k].shape[0] / module.weight.shape[0] * 100

    # Log the total percentage of frozen neurons
    return {"total": frozen_neurons / total_neurons * 100, "layer": per_layer_frozen_neurons}


class NEqScheduler(object):
    def __init__(self, model: nn.Module, lr_scheduler, velocity_momentum: float = 0.0, eps: float = 0.001):
        self._model: nn.Module = model
        self._velocity_mu: float = velocity_momentum
        self._grad_mask = {}
        self._total_neurons = 0
        self._eps = eps
        self._hooks: dict[str, NEqHook] = self._attach_hooks()
        self._lr_scheduler = lr_scheduler
        self._set_active(False)

    def __del__(self):
        self._set_active(False)

    def __enter__(self):
        self._set_active(True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._set_active(False)

    def _set_active(self, active: bool):
        for h in self._hooks:
            self._hooks[h].set_active(active)

    def step(self, init_step: bool = False) -> dict | None:
        if init_step:
            for k in self._hooks:
                self._hooks[k].step()
            return None

        frozen_neurons = _log_masks(self._model, self._grad_mask, self._total_neurons)

        for k in self._hooks:
            self._hooks[k].step()
            self._compute_masks(k, self._hooks[k].velocity)

        if self._lr_scheduler:
            self._lr_scheduler.step()

        return frozen_neurons

    def _compute_masks(self, k, velocity):
        if velocity is not None:
            # How many neurons to select as "to freeze" as percentage of the total number of neurons
            self._grad_mask[k] = torch.where(torch.abs(velocity.detach().clone()) < self._eps)[0]
        else:
            self._grad_mask[k] = torch.tensor([])
        self._grad_mask[k] = self._grad_mask[k].to(torch.long)

    def _attach_hooks(self) -> dict[str, NEqHook]:
        hooks = {}
        for n, m in self._model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
                hooks[n] = NEqHook(n, m, momentum=self._velocity_mu)
        print(f"Initialized {len(hooks)} hooks.")

        self._total_neurons = 0

        for m in self._model.modules():
            if isinstance(m, nn.Linear):
                self._total_neurons += m.weight.shape[0]
            if isinstance(m, nn.Conv2d):
                self._total_neurons += m.weight.shape[0]
            if isinstance(m, nn.BatchNorm2d):
                self._total_neurons += m.weight.shape[0]
        print(f"Total Neurons in the model: {self._total_neurons}")
        return hooks
