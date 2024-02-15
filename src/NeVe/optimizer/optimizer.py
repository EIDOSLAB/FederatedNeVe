import copy

from NeVe.utils.hook import NeVeHook
from torch import nn


def _mse(a, b):
    return ((a - b) ** 2).mean()


def _update_mse_metrics(current_metrics: list, new_metrics: list) -> dict[str, float | list[float]]:
    result = {
        "model_metric": float("Inf"),
        "model_metric_avg": float("Inf"),
        "model_metric_mse": None,
        "layers_metric_mse": None
    }

    # Update velocities if vector 'new_velocities' is not null
    if new_metrics is not None:
        result["model_metric"] = sum([sum(abs(velocities)) for velocities in new_metrics])
        result["model_metric_avg"] = result["model_metric"] / sum([len(vals) for vals in new_metrics])

    if not (current_metrics is None or new_metrics is None
            or len(current_metrics) != len(new_metrics)
            or len(current_metrics) == 0):
        mses = [_mse(val1, val2) for val1, val2 in zip(current_metrics, new_metrics)]
        result["model_metric_mse"] = sum(mses) / len(mses)
        result["layers_metric_mse"] = mses
    return result


class NeVeOptimizer(object):
    def __init__(self, model: nn.Module, velocity_momentum: float = 0.5, stop_threshold: float = 0.001):
        self._model: nn.Module = model
        self._velocity_mu: float = velocity_momentum
        self._stop_threshold: float = stop_threshold
        self._hooks: dict[str, NeVeHook] = self._attach_hooks()
        self._velocity_cache: list = []
        self._set_active(False)

    def __enter__(self):
        self._set_active(True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._set_active(False)

    def _set_active(self, active: bool):
        for h in self._hooks:
            self._hooks[h].set_active(active)

    def step(self, init_step: bool = False) -> dict[str, dict | bool]:
        if init_step:
            for k in self._hooks:
                self._hooks[k].reset()
            return {}
        data = {
            "neve": {},
            "velocity": {},
            "continue_training": True
        }
        neve_new_metrics = []
        for k in self._hooks:
            # Get layers velocity from the hooks
            velocity = copy.deepcopy(self._hooks[k].get_velocity().detach().cpu())
            # Log velocities histogram
            data["velocity"][f"{k}"] = velocity
            # Save this epoch velocities for the next iteration
            neve_new_metrics.append(velocity)
            self._hooks[k].reset()

        # Evaluate the velocities mse
        mse_data = _update_mse_metrics(self._velocity_cache, neve_new_metrics)
        self._velocity_cache = neve_new_metrics

        # Log velocity
        data["neve"]["model_value"] = mse_data["model_metric"]
        data["neve"]["model_avg_value"] = mse_data["model_metric_avg"]
        # Log model overall mse velocity
        if mse_data["model_metric_mse"] is not None:
            data["neve"]["model_mse_value"] = mse_data["model_metric_mse"]
        # Log for each layer the overall mse velocity
        if mse_data["layers_metric_mse"] is not None:
            data["neve"]["layers_mse_value"] = {}
            for count, layer_value in enumerate(mse_data["layers_metric_mse"]):
                data["neve"]["layers_mse_value"][str(count)] = layer_value
        # Log stop criterion; we can perform early-stop if the current velocity is below a certain threshold
        if mse_data["model_metric_avg"] < self._stop_threshold:
            data["continue_training"] = False
        return data

    def _attach_hooks(self) -> dict[str, NeVeHook]:
        hooks = {}
        for n, m in self._model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
                hooks[n] = NeVeHook(n, m, momentum=self._velocity_mu)
        print(f"Initialized {len(hooks)} hooks.")
        return hooks
