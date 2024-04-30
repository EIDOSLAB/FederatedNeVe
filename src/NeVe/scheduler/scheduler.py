from torch import nn

from NeVe.utils import NeVeData, NeVeHook


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


class NeVeScheduler(object):
    def __init__(self, model: nn.Module, lr_scheduler, velocity_momentum: float = 0.5, stop_threshold: float = 0.001,
                 only_last_layer: bool = False):
        self._model: nn.Module = model
        self._velocity_mu: float = velocity_momentum
        self._stop_threshold: float = stop_threshold
        self._hooks: dict[str, NeVeHook] = self._attach_hooks(only_last_layer=only_last_layer)
        self._set_active(False)
        self._velocity_cache: list = []
        self._lr_scheduler = lr_scheduler

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

    def step(self, init_step: bool = False) -> NeVeData | None:
        if init_step:
            for k in self._hooks:
                self._hooks[k].reset()
            return None
        data = NeVeData()
        neve_new_metrics = []
        for k in self._hooks:
            # Get layers velocity from the hooks
            velocity = self._hooks[k].get_velocity().detach().cpu()
            # Log velocities histogram
            data.add_velocity(k, velocity)
            # Save this epoch velocities for the next iteration
            neve_new_metrics.append(velocity)
            self._hooks[k].reset()

        # Evaluate the velocities mse
        mse_data = _update_mse_metrics(self._velocity_cache, neve_new_metrics)

        self._velocity_cache = neve_new_metrics
        data.update_velocities(mse_data, self._stop_threshold)
        # Update scheduler
        if self._lr_scheduler and data.mse_velocity:
            self._lr_scheduler.step(data.mse_velocity)
        return data

    def _attach_hooks(self, only_last_layer: bool = False) -> dict[str, NeVeHook]:
        hooks = {}
        # Only attach hooks to the last layer of the model
        if only_last_layer:
            ll_name, ll_module = None, None
            for n, m in self._model.named_modules():
                if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
                    ll_name, ll_module = n, m
            for n, m in self._model.named_modules():
                if n == ll_name and m == ll_module and isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
                    hooks[n] = NeVeHook(n, m, momentum=self._velocity_mu)
        # Attach hooks to all layers of the model
        else:
            for n, m in self._model.named_modules():
                if isinstance(m, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
                    hooks[n] = NeVeHook(n, m, momentum=self._velocity_mu)
        print(f"Initialized {len(hooks)} hooks.")
        return hooks
