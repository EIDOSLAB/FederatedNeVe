class NeVeData:
    def __init__(self):
        self._data = {
            "neve": {},
            "velocity": {},
            "neurons_velocity": {},
            "continue_training": True
        }

    @property
    def as_dict(self):
        return self._data

    @property
    def continue_training(self):
        return self._data["continue_training"]

    @property
    def neurons_velocity(self):
        return self._data["neurons_velocity"]

    @property
    def mse_velocity(self):
        if "model_mse_value" in self._data["neve"]:
            return self._data["neve"]["model_mse_value"]
        return None

    def add_velocity(self, k, velocity):
        global_velocity, neurons_velocity = velocity
        self._data["velocity"][f"{k}"] = global_velocity
        self._data["neurons_velocity"][f"{k}"] = neurons_velocity

    def update_velocities(self, mse_data: dict, stop_threshold: float):
        for key in ["model_metric", "model_metric_avg", "model_metric_mse", "layers_metric_mse"]:
            assert key in mse_data, f"'{key}' not found in mse_data keys."
        assert stop_threshold > 0.0, "stop_threshold must be > 0"

        # Update neve values
        self._data["neve"]["model_value"] = mse_data["model_metric"]
        self._data["neve"]["model_avg_value"] = mse_data["model_metric_avg"]
        # Log model overall mse velocity
        if mse_data["model_metric_mse"] is not None:
            self._data["neve"]["model_mse_value"] = mse_data["model_metric_mse"]
        # Log for each layer their overall mse velocity
        if mse_data["layers_metric_mse"] is not None:
            self._data["neve"]["layers_mse_value"] = {}
            for count, layer_value in enumerate(mse_data["layers_metric_mse"]):
                self._data["neve"]["layers_mse_value"][str(count)] = layer_value
        # Log stop criterion; we can perform early-stop if the current velocity is below a certain threshold
        if mse_data["model_metric_avg"] < stop_threshold:
            self._data["continue_training"] = False
