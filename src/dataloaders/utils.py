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


def prepare_data(train_set: Dataset, test_set: Dataset, aux_set: Dataset | None, val_percentage: int = 10,
                 split_iid: bool = True,
                 num_clients: int = 1, concentration: float = 0.5, seed: int = 42, batch_size: int = 32):
    if split_iid:
        train_l, test_l, val_l, train_distribution = split_data_iid(train_set, test_set, val_percentage,
                                                                    num_clients, seed, batch_size)
    else:
        train_l, test_l, val_l, train_distribution = split_data_not_iid(train_set, test_set, val_percentage,
                                                                        num_clients, concentration, seed, batch_size)
    aux_l = None
    if aux_set:
        aux_l = DataLoader(aux_set, batch_size=batch_size, shuffle=False)
    return train_l, test_l, val_l, aux_l, train_distribution


def split_data_iid(train_set: Dataset, test_set: Dataset, val_percentage: int = 10,
                   num_clients: int = 1, seed: int = 42, batch_size: int = 32):
    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(train_set) // num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(train_set, lengths, torch.Generator().manual_seed(seed))

    # Split each partition into train/val and create DataLoader
    train_loaders = []
    val_loaders = []
    train_distribution = []
    for ds in datasets:
        len_val = len(ds) // val_percentage  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(seed))
        train_loaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        val_loaders.append(DataLoader(ds_val, batch_size=batch_size, shuffle=False))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loaders, val_loaders, test_loader, train_distribution


def split_data_not_iid(train_set: Dataset, test_set: Dataset, val_percentage: float = 10, num_clients: int = 1,
                       concentration: float = 0.5, seed: int = 42, batch_size: int = 32):
    # SPLIT DATA INTO TRAIN AND TEST SET
    len_val = len(train_set) // val_percentage  # 10 % validation set
    len_train = len(train_set) - len_val
    lengths = [len_train, len_val]
    ds_train, ds_val = random_split(train_set, lengths, torch.Generator().manual_seed(seed))

    # TRAIN DATA MANAGEMENT
    # Transform data to numpy so that we can use the create_lda_partitions method
    train_loaders, train_distribution = _create_lda_dataloaders(ds_train, dirichlet_dist=None,
                                                                num_partitions=num_clients, concentration=concentration,
                                                                seed=seed, batch_size=batch_size, shuffle=True)
    # VALIDATION DATA MANAGEMENT
    # Transform data to numpy so that we can use the create_lda_partitions method
    val_loaders, _ = _create_lda_dataloaders(ds_val, dirichlet_dist=train_distribution,
                                             num_partitions=num_clients, concentration=concentration,
                                             seed=seed, batch_size=batch_size, shuffle=True)
    # TEST DATA MANAGEMENT
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loaders, val_loaders, test_loader, train_distribution


def _create_lda_dataloaders(dataset, dirichlet_dist=None, num_partitions: int = 1,
                            concentration: float = 0.5, seed: int = 42, batch_size: int = 1, shuffle: bool = True):
    dataset_numpy = _dataset_to_numpy(dataset, batch_size=batch_size)
    data_partitions, distribution = create_lda_partitions(dataset_numpy,
                                                          dirichlet_dist=dirichlet_dist,
                                                          num_partitions=num_partitions,
                                                          concentration=concentration,
                                                          seed=seed, accept_imbalanced=True)
    # Transform data_partitions into dataloaders
    dataloaders = []
    for partition in data_partitions:
        dataloaders.append(DataLoader(NumpyDataset(partition[0], partition[1]), batch_size=batch_size, shuffle=shuffle))
    return dataloaders, distribution


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
