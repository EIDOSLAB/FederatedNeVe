import os.path
from collections import OrderedDict

import flwr as fl
import torch
from torch.utils.data import DataLoader

from src.NeVe.federated.scheduler import FederatedNeVeScheduler
from src.NeVe.scheduler import ReduceLROnLocalPlateau
from src.models import get_model
from src.utils import get_optimizer, get_scheduler
from src.utils.trainer import run


class FederatedDefaultClient(fl.client.NumPyClient):
    def __init__(self, train_loader: DataLoader, valid_loader: DataLoader, test_loader: DataLoader,
                 dataset_name: str = "cifar10", optimizer_name: str = "sgd",
                 lr: float = 0.1, momentum: float = 0.9, weight_decay: float = 5e-4, amp: bool = True,
                 client_id: int = 0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = amp
        self.epoch = 0
        self.model = get_model(dataset=dataset_name, device=str(self.device))
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.client_id = client_id
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer = get_optimizer(self.model, opt_name=optimizer_name, starting_lr=self.lr,
                                       momentum=self.momentum, weight_decay=self.weight_decay)
        self.scheduler = get_scheduler(self.model, optimizer=self.optimizer, use_neve=False, dataset=dataset_name)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device == "cuda" and self.amp))

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # Perform one epoch of training
        training_stats = run(self.model, self.train_loader, self.optimizer, self.scaler, self.device,
                             self.amp, self.epoch, "Train")
        # Unwrap training stats
        loss = training_stats["loss"]
        accuracy_1, accuracy_5 = training_stats["accuracy"]["top1"], training_stats["accuracy"]["top5"]

        # Update scheduler
        self.epoch += 1
        if not isinstance(self.scheduler, FederatedNeVeScheduler):
            self.scheduler.step()

        # Return stats in a structured way
        results_data = {
            "loss": float(loss),
            "accuracy_top1": float(accuracy_1),
            "accuracy_top5": float(accuracy_5),
            "client_id": self.client_id,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        return self.get_parameters(config={}), len(self.train_loader), results_data

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        # Validate the model on the validation-set
        stats = run(self.model, self.valid_loader, None, self.scaler, self.device,
                    self.amp, self.epoch, "Validation")
        val_loss = stats["loss"]
        val_acc_1, val_acc_5 = stats["accuracy"]["top1"], stats["accuracy"]["top5"]

        # Validate the model on the test-set
        stats = run(self.model, self.test_loader, None, self.scaler, self.device,
                    self.amp, self.epoch, "Test")
        test_loss = stats["loss"]
        test_acc_1, test_acc_5 = stats["accuracy"]["top1"], stats["accuracy"]["top5"]

        # Return stats in a structured way
        results_data = {
            # Validation
            "val_loss": float(val_loss),
            "val_accuracy_top1": float(val_acc_1),
            "val_accuracy_top5": float(val_acc_5),
            "val_size": len(self.valid_loader),
            # Test
            "test_loss": float(test_loss),
            "test_accuracy_top1": float(test_acc_1),
            "test_accuracy_top5": float(test_acc_5),
            "test_size": len(self.test_loader),
        }
        return float(val_loss), len(self.valid_loader), results_data


class FederatedNeVeClient(FederatedDefaultClient):
    def __init__(self, train_loader: DataLoader, valid_loader: DataLoader, test_loader: DataLoader,
                 aux_loader: DataLoader,
                 dataset_name: str = "cifar10", optimizer_name: str = "sgd",
                 lr: float = 0.1, momentum: float = 0.9, weight_decay: float = 5e-4, amp: bool = True,
                 client_id: int = 0,
                 neve_momentum: float = 0.5, neve_epsilon: float = 0.001, neve_alpha: float = 0.5, neve_delta: int = 10,
                 use_disk: bool = False, neve_disk_folder: str = "../neve_data/"):
        super().__init__(train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader,
                         dataset_name=dataset_name, optimizer_name=optimizer_name,
                         lr=lr, momentum=momentum, weight_decay=weight_decay, amp=amp,
                         client_id=client_id)

        self.aux_loader = aux_loader
        self.is_neve_setupped = False
        self.continue_training = True
        self.use_disk = use_disk
        self.neve_disk_folder = neve_disk_folder
        self.scheduler = FederatedNeVeScheduler(self.model,
                                                ReduceLROnLocalPlateau(self.optimizer,
                                                                       factor=neve_alpha, patience=neve_delta),
                                                velocity_momentum=neve_momentum,
                                                stop_threshold=neve_epsilon,
                                                save_path=self.neve_disk_folder,
                                                client_id=client_id)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        if self.use_disk:
            self._load_from_disk()

        if not self.is_neve_setupped:
            # Get the velocity value before the training step (velocity at time t-1)
            with self.scheduler:
                _ = run(self.model, self.aux_loader, None, self.scaler, self.device, self.amp, self.epoch, "Aux")
            _ = self.scheduler.step(init_step=True)
            self.is_neve_setupped = True

        # Perform default fit step
        if self.continue_training:
            params, len_ds, train_logs = super().fit(parameters, config)
        else:
            return [], len(self.train_loader), {}
        # Get the velocity value after the training step (velocity at time t)
        with self.scheduler:
            _ = run(self.model, self.aux_loader, None, self.scaler, self.device, self.amp, self.epoch, "Aux")
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

        # Save states (Optimizers, schedulers, etc...)
        if self.use_disk:
            self._save_into_disk()
        return params, len_ds, train_logs

    def _get_client_checkpoint_path(self):
        return os.path.join(self.neve_disk_folder, f"client_{self.client_id}_state.pt")

    def _save_into_disk(self):
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
            self._get_client_checkpoint_path()
        )

    def _load_from_disk(self):
        if not os.path.exists(self.neve_disk_folder):
            os.makedirs(self.neve_disk_folder)
            os.makedirs(os.path.join(self.neve_disk_folder, "activations"))

        if not os.path.exists(self._get_client_checkpoint_path()):
            return

        self.scheduler.load_activations(self.device)
        checkpoint = torch.load(self._get_client_checkpoint_path())
        self.epoch = checkpoint["epoch"]
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.is_neve_setupped = checkpoint["is_neve_setupped"]
        self.continue_training = checkpoint["continue_training"]
        self.scheduler.load_state_dicts(checkpoint["neve_scheduler_state"], checkpoint["neve_velocity_cache"])
