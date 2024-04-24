import os.path

import torch
from torch.utils.data import DataLoader

from src.NeVe.federated.scheduler import FederatedNeVeScheduler
from src.NeVe.scheduler import ReduceLROnLocalPlateau
from src.my_flwr.clients.baseline_client import FederatedDefaultClient
from src.utils.trainer import run


class FederatedNeVeClient(FederatedDefaultClient):
    def __init__(self, train_loader: DataLoader, valid_loader: DataLoader, test_loader: DataLoader,
                 aux_loader: DataLoader,
                 use_groupnorm: bool = True, groupnorm_channels: int = 2,
                 model_name: str = "resnet18", device: str = "cuda",
                 dataset_name: str = "cifar10", optimizer_name: str = "sgd",
                 lr: float = 0.1, momentum: float = 0.9, weight_decay: float = 5e-4, amp: bool = True,
                 scheduler_name: str = "neve", client_id: int = 0,
                 neve_momentum: float = 0.5, neve_epsilon: float = 0.001, neve_alpha: float = 0.5, neve_delta: int = 10,
                 neve_only_last_layer: bool = False, use_disk: bool = False, disk_folder: str = "../fclients_data/"):
        super().__init__(train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
                         use_groupnorm=use_groupnorm, groupnorm_channels=groupnorm_channels,
                         model_name=model_name, device=device,
                         dataset_name=dataset_name, optimizer_name=optimizer_name,
                         lr=lr, momentum=momentum, weight_decay=weight_decay, amp=amp,
                         scheduler_name=scheduler_name, client_id=client_id,
                         use_disk=use_disk, disk_folder=disk_folder)

        self.aux_loader = aux_loader
        self.is_neve_setupped = False
        self.continue_training = True
        self.scheduler = FederatedNeVeScheduler(self.model,
                                                ReduceLROnLocalPlateau(self.optimizer,
                                                                       factor=neve_alpha, patience=neve_delta),
                                                velocity_momentum=neve_momentum,
                                                stop_threshold=neve_epsilon,
                                                save_path=self.disk_folder,
                                                only_last_layer=neve_only_last_layer,
                                                client_id=client_id)

    def __del__(self):
        print("FederatedNeVeClient -> Del")
        del self.scheduler

    def _fit_method(self, parameters, config):
        if not self.is_neve_setupped and self.scheduler:
            # Get the velocity value before the training step (velocity at time t-1)
            with self.scheduler:
                _ = run(self.model, self.aux_loader, None, self.scaler, self.device, self.amp, self.epoch, "Aux")
            _ = self.scheduler.step(init_step=True)
            self.is_neve_setupped = True

        # Perform default fit step
        if self.continue_training:
            params, len_ds, train_logs = super()._fit_method(parameters, config)
        else:
            return [], len(self.train_loader), {}
        # Get the velocity value after the training step (velocity at time t)
        with self.scheduler:
            _ = run(self.model, self.aux_loader, None, self.scaler, self.device, self.amp, self.epoch, "Aux")
        # Step the NeVe scheduler and get velocity information
        velocity_data = self.scheduler.step()
        for key, value in velocity_data.as_dict["neve"].items():
            if isinstance(value, dict):
                continue
            train_logs[f"neve.{key}"] = value.item()
        train_logs["neve.continue_training"] = velocity_data.continue_training
        if self.continue_training:
            self.continue_training = velocity_data.continue_training
        print(f"Client: {self.client_id} - Model Avg. Velocity: {train_logs['neve.model_avg_value']}")
        print(f"Client: {self.client_id} - Continue training? {self.continue_training}")
        return params, len_ds, train_logs

    def _load(self):
        if not os.path.exists(self.disk_folder):
            os.makedirs(self.disk_folder)
            os.makedirs(os.path.join(self.disk_folder, "activations"))

        if not os.path.exists(self.get_client_checkpoint_path()):
            return

        self.scheduler.load_activations(self.device)
        checkpoint = torch.load(self.get_client_checkpoint_path())
        self.epoch = checkpoint["epoch"]
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.is_neve_setupped = checkpoint["is_neve_setupped"]
        self.continue_training = checkpoint["continue_training"]
        self.scheduler.load_state_dicts(checkpoint["neve_scheduler_state"], checkpoint["neve_velocity_cache"])

    def _save(self):
        self.scheduler.save_activations()
        neve_scheduler_state, neve_velocity_cache = self.scheduler.state_dicts()
        torch.save(
            {
                "epoch": self.epoch,
                "neve_scheduler_state": neve_scheduler_state,
                "neve_velocity_cache": neve_velocity_cache,
                "optimizer_state": self.optimizer.state_dict(),
                "scaler_state": self.scaler.state_dict(),
                "is_neve_setupped": self.is_neve_setupped,
                "continue_training": self.continue_training,
            },
            self.get_client_checkpoint_path()
        )
