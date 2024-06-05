import abc
import os
from abc import ABC
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

from src.models import get_model
from src.utils import get_optimizer, get_scheduler


class AFederatedClient(ABC):

    def __init__(self, train_loader: DataLoader, valid_loader: DataLoader, test_loader: DataLoader,
                 use_groupnorm: bool = True, groupnorm_channels: int = 2,
                 model_name: str = "resnet18", device: str = "cuda",
                 dataset_name: str = "cifar10", optimizer_name: str = "sgd",
                 lr: float = 0.1, momentum: float = 0.9, weight_decay: float = 5e-4, amp: bool = True,
                 scheduler_name: str = "baseline", client_id: int = 0,
                 use_disk: bool = False, disk_folder: str = "../fclients_data/"):
        self.device = device
        self.amp = amp
        self.epoch = 0
        self.model = get_model(dataset=dataset_name, model_name=model_name, device=self.device,
                               use_groupnorm=use_groupnorm, groupnorm_channels=groupnorm_channels)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.client_id = client_id
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.use_disk = use_disk
        self.disk_folder = disk_folder
        self.optimizer = get_optimizer(self.model, opt_name=optimizer_name, starting_lr=self.lr,
                                       momentum=self.momentum, weight_decay=self.weight_decay)
        self.scheduler = get_scheduler(self.model, optimizer=self.optimizer, scheduler_name=scheduler_name,
                                       dataset=dataset_name)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device == "cuda" and self.amp))

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # Load components states (Optimizers, schedulers, etc...)
        if self.use_disk:
            self._load()
        # Perform the fit method
        params, len_ds, results_data = self._fit_method(parameters, config)
        self.epoch += 1
        # Save components states (Optimizers, schedulers, etc...)
        if self.use_disk:
            self._save()
        return params, len_ds, results_data

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # Load components states (Optimizers, schedulers, etc...)
        if self.use_disk:
            self._load()
        self._update_components()
        # Perform the eval method
        eval_loss, len_ds, results_data = self._evaluate_method(parameters, config)
        return eval_loss, len_ds, results_data

    def get_client_checkpoint_path(self):
        return os.path.join(self.disk_folder, f"client_{self.client_id}_state.pt")

    @abc.abstractmethod
    def _update_components(self):
        pass

    @abc.abstractmethod
    def _fit_method(self, parameters, config) -> tuple[list, int, dict]:
        pass

    @abc.abstractmethod
    def _evaluate_method(self, parameters, config) -> tuple[float, int, dict]:
        pass

    @abc.abstractmethod
    def _load(self):
        pass

    @abc.abstractmethod
    def _save(self):
        pass
