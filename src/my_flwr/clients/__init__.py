from torch.utils.data import DataLoader

from src.my_flwr.clients.baseline_client import FederatedDefaultClient
from src.my_flwr.clients.neve_client import FederatedNeVeClient


def get_client(train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, aux_loader: DataLoader | None,
               use_groupnorm: bool = True, groupnorm_channels: int = 2, optimizer_name: str = "sgd",
               dataset_name: str = "cifar10", lr: float = 0.1, momentum: float = 0.9, weight_decay: float = 5e-4,
               amp: bool = True, client_id: int = 0, device: str = "cuda",
               neve_momentum: float = 0.5, neve_epsilon: float = 0.001, neve_alpha: float = 0.5, neve_delta: int = 10,
               neve_only_last_layer: bool = False,
               scheduler_name: str = "baseline", use_disk: bool = False, disk_folder="../fclients_data/"):
    use_neve = scheduler_name == "neve"
    if use_neve:
        return FederatedNeVeClient(train_loader=train_loader, valid_loader=val_loader, test_loader=test_loader,
                                   aux_loader=aux_loader,
                                   use_groupnorm=use_groupnorm, groupnorm_channels=groupnorm_channels, device=device,
                                   dataset_name=dataset_name, optimizer_name=optimizer_name,
                                   lr=lr, momentum=momentum, weight_decay=weight_decay, amp=amp,
                                   neve_epsilon=neve_epsilon, neve_momentum=neve_momentum,
                                   neve_alpha=neve_alpha, neve_delta=neve_delta,
                                   neve_only_last_layer=neve_only_last_layer,
                                   client_id=client_id, use_disk=use_disk, disk_folder=disk_folder)
    else:
        return FederatedDefaultClient(train_loader=train_loader, valid_loader=val_loader, test_loader=test_loader,
                                      use_groupnorm=use_groupnorm, groupnorm_channels=groupnorm_channels, device=device,
                                      dataset_name=dataset_name, optimizer_name=optimizer_name,
                                      lr=lr, momentum=momentum, weight_decay=weight_decay, amp=amp,
                                      client_id=client_id, use_disk=use_disk, disk_folder=disk_folder)
