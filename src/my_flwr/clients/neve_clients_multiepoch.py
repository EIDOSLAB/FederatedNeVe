from torch.utils.data import DataLoader

from src.my_flwr.clients.neve_client import FederatedNeVeClient
from src.utils.trainer import run


class FederatedNeVeMultiEpochClient(FederatedNeVeClient):
    def __init__(self, train_loader: DataLoader, valid_loader: DataLoader, test_loader: DataLoader,
                 aux_loader: DataLoader,
                 use_groupnorm: bool = True, groupnorm_channels: int = 2,
                 use_pretrain: bool = False,
                 model_name: str = "resnet18", device: str = "cuda",
                 dataset_name: str = "cifar10", optimizer_name: str = "sgd",
                 lr: float = 0.1, momentum: float = 0.9, weight_decay: float = 5e-4, amp: bool = True,
                 scheduler_name: str = "neve", client_id: int = 0,
                 min_lr: float = 0.00001,
                 neve_use_lr_scheduler: bool = True, neve_use_early_stop: bool = False,
                 neve_momentum: float = 0.5, neve_epsilon: float = 0.001, neve_alpha: float = 0.5, neve_delta: int = 10,
                 neve_only_last_layer: bool = False, num_train_epochs: int = 1,
                 use_disk: bool = False, disk_folder: str = "../fclients_data/"):
        super().__init__(train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
                         aux_loader=aux_loader,
                         use_groupnorm=use_groupnorm, groupnorm_channels=groupnorm_channels,
                         use_pretrain=use_pretrain,
                         model_name=model_name, device=device,
                         dataset_name=dataset_name, optimizer_name=optimizer_name,
                         lr=lr, momentum=momentum, weight_decay=weight_decay, amp=amp,
                         min_lr=min_lr,
                         scheduler_name=scheduler_name, client_id=client_id,
                         neve_use_lr_scheduler=neve_use_lr_scheduler, neve_use_early_stop=neve_use_early_stop,
                         neve_momentum=neve_momentum, neve_epsilon=neve_epsilon, neve_alpha=neve_alpha,
                         neve_delta=neve_delta, neve_only_last_layer=neve_only_last_layer,
                         use_disk=use_disk, disk_folder=disk_folder)
        self.num_train_epochs = num_train_epochs

    def _fit_method(self, parameters, config) -> tuple[list, int, dict]:
        if not self.is_neve_setupped and self.neve_scheduler:
            # Get the velocity value before the training step (velocity at time t-1)
            with self.neve_scheduler:
                _ = run(self.model, self.aux_loader, None, self.scaler, self.device, self.amp, self.epoch, "Aux")
            _ = self.neve_scheduler.step(init_step=True)
            self.is_neve_setupped = True

        params, len_ds, train_logs = [], len(self.train_loader), {}
        for epoch in range(self.num_train_epochs):
            # Perform default fit step
            if self.use_early_stop and not self.continue_training:
                return [], len(self.train_loader), {}
            else:
                params, len_ds, train_logs = super()._fit_method(parameters, config)
            # Get the velocity value after the training step (velocity at time t)
            with self.neve_scheduler:
                _ = run(self.model, self.aux_loader, None, self.scaler, self.device, self.amp, self.epoch, "Aux")
            # Step the NeVe scheduler and get velocity information
            velocity_data = self.neve_scheduler.step()
            for key, value in velocity_data.as_dict["neve"].items():
                if isinstance(value, dict):
                    continue
                train_logs[f"epoch_{epoch}.neve.{key}"] = value.item()
                train_logs[f"neve.{key}"] = value.item()
            train_logs[f"epoch_{epoch}.neve.continue_training"] = velocity_data.continue_training
            if self.continue_training:
                self.continue_training = velocity_data.continue_training
            print(f"Client: {self.client_id} - Model Avg. Velocity: "
                  f"{train_logs[f'epoch_{epoch}.neve.model_avg_value']}")
            print(f"Client: {self.client_id} - Continue training? {self.continue_training}")
        return params, len_ds, train_logs
