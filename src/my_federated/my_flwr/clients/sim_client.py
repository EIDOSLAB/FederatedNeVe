from collections import OrderedDict

import flwr as fl
import torch
from torch.utils.data import DataLoader

from my_federated.NeVe.scheduler import NeVeScheduler
from my_federated.datasets.dataloader.loader import load_partition, get_dataset_fed_path, load_loader, \
    load_transform
from my_federated.models import get_model
from my_federated.utils import get_optimizer, get_scheduler
from my_federated.utils.metrics import get_label_distribution
from my_federated.utils.trainer import train_model, eval_model


class SimulationNeVeClient(fl.client.NumPyClient):
    def __init__(self, dataset_root: str, dataset_name: str = "cifar10", aux_loader: DataLoader | None = None,
                 use_groupnorm: bool = True, groupnorm_channels: int = 2, optimizer_name: str = "sgd",
                 scheduler_name: str = "constant",
                 use_pretrain: bool = False, inner_epochs: int = 1,
                 lr: float = 0.1, momentum: float = 0.9, weight_decay: float = 5e-4,
                 amp: bool = True, client_id: int = 0, model_name: str = "resnet18", device: str = "cuda",
                 neve_momentum: float = 0.5, neve_only_last_layer: bool = False,
                 dataset_iid: bool = True, num_clients: int = 10, lda_concentration: float = 0.1, seed: int = 42,
                 batch_size: int = 32, medmnist_size: int = 224, strategy_name: str = "fedavg", data_distribution=None,
                 val_percentage: int = 10, dataset_task: str = "multi-class", pin_data_in_memory: bool = False):
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name
        self.device = device
        self.amp = amp
        self.epoch = 0
        self.num_classes = 0
        self.model_name = model_name
        self.use_groupnorm = use_groupnorm
        self.groupnorm_channels = groupnorm_channels
        self.use_pretrain = use_pretrain
        self.dataset_iid = dataset_iid
        self.num_clients = num_clients
        self.lda_concentration = lda_concentration
        self.seed = seed
        self.medmnist_size = medmnist_size
        self.batch_size = batch_size
        self.aux_loader = aux_loader
        self.client_id = client_id
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.scheduler_name = scheduler_name
        self.strategy_name = strategy_name
        self.data_distribution = data_distribution
        self.val_percentage = val_percentage
        self.dataset_task = dataset_task
        self.pin_data_in_memory = pin_data_in_memory
        self.train_loader = None
        self.valid_loader = None

        self.dataset_fed_path = get_dataset_fed_path(self.dataset_root, self.dataset_name,
                                                     medmnist_size=self.medmnist_size, seed=self.seed,
                                                     val_size=self.val_percentage / 100,
                                                     num_clients=self.num_clients, ds_iid=self.dataset_iid,
                                                     strategy_name=strategy_name)

        self.inner_epochs = inner_epochs
        if self.inner_epochs <= 0:
            self.inner_epochs = 1
        self.neve_momentum = neve_momentum
        self.neve_only_last_layer = neve_only_last_layer

    def get_parameters(self, config):
        model, self.num_classes = get_model(dataset=self.dataset_name, model_name=self.model_name,
                                            device=self.device, use_pretrain=self.use_pretrain,
                                            use_groupnorm=self.use_groupnorm,
                                            groupnorm_channels=self.groupnorm_channels)
        return self._get_model_parameters(model)

    @staticmethod
    def _get_model_parameters(model):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters, config):
        model, self.num_classes = get_model(dataset=self.dataset_name, model_name=self.model_name,
                                            device=self.device, use_pretrain=self.use_pretrain,
                                            use_groupnorm=self.use_groupnorm,
                                            groupnorm_channels=self.groupnorm_channels)
        # Update parameters BEFORE creating the optimizer
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        # Create optimizer, scheduler, etc...
        optimizer = get_optimizer(model, opt_name=self.optimizer_name, starting_lr=self.lr,
                                  momentum=self.momentum, weight_decay=self.weight_decay)
        scheduler = get_scheduler(optimizer=optimizer, scheduler_name=self.scheduler_name)
        neve_scheduler = NeVeScheduler(model, velocity_momentum=self.neve_momentum,
                                       only_last_layer=self.neve_only_last_layer)
        scaler = torch.amp.GradScaler("cuda", enabled=(self.device == "cuda" and self.amp))

        if not self.pin_data_in_memory:
            transforms = load_transform(self.dataset_name, self.model_name)
            partition_data = load_partition(self.dataset_fed_path, self.client_id,
                                            partitions=config.get("partitions_2_load"), transform_2_pil=True)
            loaders = {}
            for loader_name, data in partition_data.items():
                if loader_name == "train":
                    data.transform = transforms[loader_name]
                    loaders[loader_name] = load_loader(data, batch_size=self.batch_size)
                else:
                    data.transform = transforms["test"]
                    loaders[loader_name] = load_loader(data, batch_size=self.batch_size, shuffle=False)
        else:
            if self.train_loader is None or self.valid_loader is None:
                transforms = load_transform(self.dataset_name, self.model_name)
                partition_data = load_partition(self.dataset_fed_path, self.client_id,
                                                partitions=config.get("partitions_2_load"), transform_2_pil=True)
                loaders = {}
                for loader_name, data in partition_data.items():
                    if loader_name == "train":
                        data.transform = transforms[loader_name]
                        self.train_loader = load_loader(data, batch_size=self.batch_size)
                        loaders[loader_name] = self.train_loader
                    else:
                        data.transform = transforms["test"]
                        self.valid_loader = load_loader(data, batch_size=self.batch_size, shuffle=False)
                        loaders[loader_name] = self.valid_loader
            else:
                loaders = {"train": self.train_loader, "validation": self.valid_loader}

        if config and scheduler:
            self.epoch = config.get('round', 0)
            print(f"Current server_round: {self.epoch}")
            for server_round in range(self.epoch):
                scheduler.step()
        return model, optimizer, scaler, neve_scheduler, loaders

    def fit(self, parameters, config):
        config["partitions_2_load"] = ["train"]
        model, optimizer, scaler, neve_scheduler, loaders = self.set_parameters(parameters, config)
        train_loader = loaders.get("train", None)
        assert train_loader is not None, "sim_client train_loader is None!"
        if self.data_distribution is None:
            self.data_distribution = get_label_distribution(train_loader, self.num_classes, self.dataset_task)
        neve_scheduler.update_label_distribution(self.data_distribution)
        # Perform the fit method
        # Fully reset NeVe stats
        neve_scheduler.full_reset()
        # Get the velocity value before the training step (velocity at time t-1)
        with neve_scheduler:
            _ = eval_model(model, self.aux_loader, self.dataset_task, self.device, self.amp,
                           epoch=self.epoch, run_type="Aux")
        _ = neve_scheduler.step(init_step=True)

        # Perform default fit step
        fit_logs = {}
        for epoch in range(self.inner_epochs):
            training_stats = train_model(model, train_loader, self.dataset_task, optimizer, scaler,
                                         self.device, self.amp, epoch=self.epoch)
            # Unwrap training stats
            loss = training_stats["loss"]
            accuracy_1 = training_stats["accuracy"]["top1"]

            # Return stats in a structured way
            fit_logs = {
                "loss": float(loss),
                "accuracy_top1": float(accuracy_1),
                "size": len(train_loader.dataset),
                "client_id": self.client_id,
                "lr": optimizer.param_groups[0]["lr"],
            }

        # Get the velocity value after the training step (velocity at time t)
        with neve_scheduler:
            _ = eval_model(model, self.aux_loader, self.dataset_task, self.device, self.amp,
                           epoch=self.epoch, run_type="Aux")
        # Step the NeVe scheduler and get velocity information
        velocity_data = neve_scheduler.step()
        #for key, value in velocity_data.as_dict["neve"].items():
        #    if isinstance(value, dict):
        #        continue
        #    fit_logs[f"neve.{key}"] = value.item()
        fit_logs["neve.velocity"] = velocity_data.velocity  # fit_logs["neve.model_avg_value"]
        print(f"Client: {self.client_id} - Model Avg. Velocity: {fit_logs['neve.velocity']}")
        #for key, value in velocity_data.velocity_hist["velocity"].items():
        #    for idx, val in enumerate(value):
        #        fit_logs[f"neve.velocity_histogram.{key}.{idx}"] = float(val)
        fit_logs["neve.client"] = self.client_id
        #
        return self._get_model_parameters(model), len(train_loader.dataset), fit_logs

    def evaluate(self, parameters, config):
        config["partitions_2_load"] = ["validation"]
        model, optimizer, scaler, neve_scheduler, loaders = self.set_parameters(parameters, config)
        valid_loader = loaders.get("validation")
        assert valid_loader is not None, "sim_client valid_loader is None!"

        # Validate the model on the validation-set
        val_stats = eval_model(model, valid_loader, self.dataset_task, self.device, self.amp,
                               epoch=self.epoch, run_type="Validation")
        val_loss, val_acc_1 = val_stats["loss"], val_stats["accuracy"]["top1"]

        # Return stats in a structured way
        eval_logs = {
            # Validation
            "loss": float(val_loss),
            "accuracy_top1": float(val_acc_1),
            "size": len(valid_loader.dataset),
            # Other info
            "client_id": self.client_id,
        }
        #
        return float(val_loss), len(valid_loader.dataset), eval_logs
