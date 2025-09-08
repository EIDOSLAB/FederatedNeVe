class NeVeData:
    def __init__(self):
        self._data = {
            "neve": {},
            "velocity": {}
        }
        self._hist_data = {
            "velocity": {}
        }

    @property
    def as_dict(self):
        return self._data

    @property
    def mse_velocity(self):
        if "model_mse_value" in self._data["neve"]:
            return self._data["neve"]["model_mse_value"]
        return None

    @property
    def velocity(self):
        if "model_avg_value" in self._data["neve"]:
            return self._data["neve"]["model_avg_value"]
        return None

    @property
    def velocity_hist(self):
        return self._hist_data

    def add_velocity(self, k, velocity):
        self._data["velocity"][f"{k}"] = velocity

    def add_velocity_hist(self, k, velocity_hist):
        self._hist_data["velocity"][f"{k}"] = velocity_hist

    def update_velocities(self, mse_data: dict):
        for key in ["model_metric", "model_metric_avg", "model_metric_mse", "layers_metric_mse"]:
            assert key in mse_data, f"'{key}' not found in mse_data keys."

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
