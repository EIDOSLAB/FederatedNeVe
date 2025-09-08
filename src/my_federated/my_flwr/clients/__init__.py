from torch.utils.data import DataLoader

from my_federated.my_flwr.clients.sim_client import SimulationNeVeClient


def get_simulation_client(dataset_root: str, dataset_name: str = "cifar10", aux_loader: DataLoader | None = None,
                          use_groupnorm: bool = True, groupnorm_channels: int = 2, optimizer_name: str = "sgd",
                          scheduler_name: str = "constant",
                          use_pretrain: bool = False, inner_epochs: int = 1,
                          dataset_iid: bool = False, num_clients: int = 10, lda_concentration: float = 0.1,
                          seed: int = 42, batch_size: int = 32,
                          lr: float = 0.1, momentum: float = 0.9, weight_decay: float = 5e-4,
                          amp: bool = True, client_id: int = 0, model_name: str = "resnet18", device: str = "cuda",
                          neve_momentum: float = 0.5, neve_only_last_layer: bool = False,
                          medmnist_size: int = 224, strategy_name: str = "fedavg",
                          data_distribution=None, val_percentage: int = 10, dataset_task: str = "multi-class",
                          pin_data_in_memory: bool = False):
    return SimulationNeVeClient(dataset_root, dataset_name, aux_loader,
                                use_groupnorm, groupnorm_channels, optimizer_name, scheduler_name,
                                use_pretrain, inner_epochs,
                                lr, momentum, weight_decay,
                                amp, client_id, model_name, device,
                                neve_momentum, neve_only_last_layer,
                                dataset_iid, num_clients, lda_concentration, seed,
                                batch_size, medmnist_size, strategy_name=strategy_name,
                                data_distribution=data_distribution,
                                val_percentage=val_percentage, dataset_task=dataset_task,
                                pin_data_in_memory=pin_data_in_memory)
