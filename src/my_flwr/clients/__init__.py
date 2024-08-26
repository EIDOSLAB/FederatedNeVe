from torch.utils.data import DataLoader

from my_flwr.clients.neve_clients_multiepoch import FederatedNeVeMultiEpochClient
from src.my_flwr.clients.baseline_client import FederatedDefaultClient
from src.my_flwr.clients.neve_client import FederatedNeVeClient


def get_client(train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, aux_loader: DataLoader | None,
               use_groupnorm: bool = True, groupnorm_channels: int = 2, optimizer_name: str = "sgd",
               use_pretrain: bool = False,
               dataset_name: str = "cifar10", lr: float = 0.1, min_lr: float = 0.00001, momentum: float = 0.9,
               weight_decay: float = 5e-4,
               amp: bool = True, client_id: int = 0, model_name: str = "resnet18", device: str = "cuda",
               use_neve: bool = False, use_neve_multiepoch: bool = False, neve_multiepoch_epochs: int = 2,
               neve_use_lr_scheduler: bool = True, neve_use_early_stop: bool = False,
               neve_momentum: float = 0.5, neve_epsilon: float = 0.001, neve_alpha: float = 0.5, neve_delta: int = 10,
               neve_only_last_layer: bool = False,
               scheduler_name: str = "baseline", use_disk: bool = False, disk_folder="../fclients_data/",
               leaf_input_dim: int = 10):
    if use_neve:
        if use_neve_multiepoch:
            return FederatedNeVeMultiEpochClient(
                train_loader=train_loader, valid_loader=val_loader, test_loader=test_loader,
                aux_loader=aux_loader,
                use_groupnorm=use_groupnorm, groupnorm_channels=groupnorm_channels,
                use_pretrain=use_pretrain,
                model_name=model_name, device=device,
                scheduler_name=scheduler_name,
                dataset_name=dataset_name, optimizer_name=optimizer_name,
                lr=lr, momentum=momentum, weight_decay=weight_decay, amp=amp,
                min_lr=min_lr,
                num_train_epochs=neve_multiepoch_epochs,
                neve_use_lr_scheduler=neve_use_lr_scheduler,
                neve_use_early_stop=neve_use_early_stop,
                neve_epsilon=neve_epsilon, neve_momentum=neve_momentum,
                neve_alpha=neve_alpha, neve_delta=neve_delta,
                neve_only_last_layer=neve_only_last_layer,
                client_id=client_id, use_disk=use_disk, disk_folder=disk_folder, leaf_input_dim=leaf_input_dim)
        else:
            return FederatedNeVeClient(train_loader=train_loader, valid_loader=val_loader, test_loader=test_loader,
                                       aux_loader=aux_loader,
                                       use_groupnorm=use_groupnorm, groupnorm_channels=groupnorm_channels,
                                       use_pretrain=use_pretrain,
                                       model_name=model_name, device=device,
                                       scheduler_name=scheduler_name,
                                       dataset_name=dataset_name, optimizer_name=optimizer_name,
                                       lr=lr, momentum=momentum, weight_decay=weight_decay, amp=amp,
                                       min_lr=min_lr,
                                       neve_use_lr_scheduler=neve_use_lr_scheduler,
                                       neve_use_early_stop=neve_use_early_stop,
                                       neve_epsilon=neve_epsilon, neve_momentum=neve_momentum,
                                       neve_alpha=neve_alpha, neve_delta=neve_delta,
                                       neve_only_last_layer=neve_only_last_layer,
                                       client_id=client_id, use_disk=use_disk, disk_folder=disk_folder,
                                       leaf_input_dim=leaf_input_dim)
    else:
        return FederatedDefaultClient(train_loader=train_loader, valid_loader=val_loader, test_loader=test_loader,
                                      use_groupnorm=use_groupnorm, groupnorm_channels=groupnorm_channels,
                                      use_pretrain=use_pretrain,
                                      model_name=model_name, device=device,
                                      scheduler_name=scheduler_name,
                                      min_lr=min_lr,
                                      dataset_name=dataset_name, optimizer_name=optimizer_name,
                                      lr=lr, momentum=momentum, weight_decay=weight_decay, amp=amp,
                                      client_id=client_id, use_disk=use_disk, disk_folder=disk_folder,
                                      leaf_input_dim=leaf_input_dim)
