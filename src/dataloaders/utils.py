import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from src.dataloaders.flwr_fedavgm_common import create_lda_partitions
from src.dataloaders.random_dataset import RandomDataset


class NumpyDataset(Dataset):
    def __init__(self, x, y):
        self.X = x
        self.Y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.Y[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def prepare_data(ds_root: str, ds_name: str,
                 train_set: Dataset, test_set: Dataset, aux_set: Dataset | None, val_percentage: int = 10,
                 split_iid: bool = True,
                 num_clients: int = 1, concentration: float = 0.5, seed: int = 42, batch_size: int = 32,
                 current_client: int = 0):
    if split_iid:
        train_l, test_l, val_l = split_data_iid(train_set, test_set, val_percentage,
                                                num_clients, seed, batch_size, current_client)
    else:
        train_l, test_l, val_l = split_data_not_iid(ds_root, ds_name,
                                                    train_set, test_set, val_percentage,
                                                    num_clients, concentration, seed, batch_size, current_client)
    aux_l = None
    if aux_set:
        aux_l = DataLoader(aux_set, batch_size=batch_size, shuffle=False)
    return train_l, test_l, val_l, aux_l


def split_data_iid(train_set: Dataset, test_set: Dataset, val_percentage: int = 10,
                   num_clients: int = 1, seed: int = 42, batch_size: int = 32, current_client: int = 0):
    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(train_set) // num_clients
    remainder = len(train_set) % num_clients
    lengths = [partition_size] * num_clients
    lengths[0] += remainder
    datasets = random_split(train_set, lengths, torch.Generator().manual_seed(seed))

    # Split partition into train/val and create DataLoader
    current_client = current_client % num_clients
    ds = datasets[current_client]
    len_val = len(ds) // val_percentage  # 10 % validation set
    len_train = len(ds) - len_val
    lengths = [len_train, len_val]
    ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(seed))
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def split_data_not_iid(ds_root: str, ds_name: str,
                       train_set: Dataset, test_set: Dataset, val_percentage: float = 10, num_clients: int = 1,
                       concentration: float = 0.5, seed: int = 42, batch_size: int = 32, current_client: int = 0):
    non_iid_path = os.path.join(ds_root, "non_iid", ds_name, f"seed_{str(seed)}", f"beta_{str(concentration)}")
    # If data is not in disk, we create the partitions and save them into disk
    if not os.path.exists(non_iid_path):
        print("Preparing non-IID partitions...")
        # SPLIT DATA INTO TRAIN AND TEST SET
        len_val = len(train_set) // val_percentage  # 10 % validation set
        len_train = len(train_set) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(train_set, lengths, torch.Generator().manual_seed(seed))

        # TRAIN DATA MANAGEMENT
        # Transform data to numpy so that we can use the create_lda_partitions method
        train_partitions, train_distribution = _create_lda_partitions(ds_train, dirichlet_dist=None,
                                                                      num_partitions=num_clients,
                                                                      concentration=concentration,
                                                                      seed=seed, batch_size=batch_size)
        # VALIDATION DATA MANAGEMENT
        # Transform data to numpy so that we can use the create_lda_partitions method
        val_partitions, _ = _create_lda_partitions(ds_val, dirichlet_dist=train_distribution,
                                                   num_partitions=num_clients, concentration=concentration,
                                                   seed=seed, batch_size=batch_size)
        os.makedirs(non_iid_path, exist_ok=True)
        for partition_idx, (train_partition, val_partition) in enumerate(zip(train_partitions, val_partitions)):
            _save_lda_partition(non_iid_path, partition_idx, train_partition, val_partition)
        print("Finished preparing non-IID partitions")
    # Now we can read the lda partitions
    train_loader, val_loader = _load_lda_partition(non_iid_path, num_partitions=num_clients, batch_size=batch_size,
                                                   current_client=current_client)
    # TEST DATA MANAGEMENT
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def _create_lda_partitions(dataset, dirichlet_dist=None, num_partitions: int = 1,
                           concentration: float = 0.5, seed: int = 42, batch_size: int = 1):
    dataset_numpy = _dataset_to_numpy(dataset, batch_size=batch_size)
    data_partitions, distribution = create_lda_partitions(dataset_numpy,
                                                          dirichlet_dist=dirichlet_dist,
                                                          num_partitions=num_partitions,
                                                          concentration=concentration,
                                                          seed=seed, accept_imbalanced=True)
    return data_partitions, distribution


def _save_lda_partition(save_path: str, partition_idx: int, train_partition, val_partition):
    file_path = os.path.join(save_path, f"{partition_idx}.npz")
    np.savez(file_path,
             train_data=train_partition[0], train_labels=train_partition[1],
             val_data=val_partition[0], val_labels=val_partition[1])


def _load_lda_partition(load_path: str, num_partitions: int = 1, batch_size: int = 1, current_client: int = 0):
    # Check the partition found in given path
    current_client = current_client % num_partitions
    partition_path = os.path.join(load_path, f"{current_client}.npz")
    if not os.path.exists(partition_path):
        raise Exception("_load_lda_partitions -> partition not found!")
    # Read data partition from disk and transform into a dataloader
    data = np.load(partition_path)
    train_dataloader = DataLoader(NumpyDataset(data["train_data"], data["train_labels"]),
                                  batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(NumpyDataset(data["val_data"], data["val_labels"]),
                                batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader


def _dataset_to_numpy(dataset, batch_size: int = 1):
    # Lista per memorizzare immagini e etichette
    all_images = []
    all_labels = []

    # Carica i dati in batch
    dataloader = DataLoader(dataset, batch_size=batch_size)
    for data in dataloader:
        images, labels = data
        all_images.append(images.numpy())
        all_labels.append(labels.numpy())

    # Concatenare tutti i batch
    x = np.concatenate(all_images, axis=0)
    y = np.concatenate(all_labels, axis=0)
    return x, y


def load_aux_dataset(shape: tuple[int, int, int] = (3, 32, 32), aux_seed: int = 42, number_samples: int = 10,
                     batch_size: int = 32, transform=None):
    aux_set = RandomDataset(img_shape=shape, dataset_size=number_samples, seed=aux_seed, transform=transform)
    aux_loader = DataLoader(aux_set, batch_size=batch_size, shuffle=False)
    return aux_loader
