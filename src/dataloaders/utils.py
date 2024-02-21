import torch
from torch.utils.data import Dataset, DataLoader, random_split


def prepare_data(train_set: Dataset, test_set: Dataset, aux_set: Dataset, val_percentage: int = 10,
                 num_clients: int = 1, seed: int = 42, batch_size: int = 32):
    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(train_set) // num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(train_set, lengths, torch.Generator().manual_seed(seed))

    # Split each partition into train/val and create DataLoader
    train_loaders = []
    val_loaders = []
    for ds in datasets:
        len_val = len(ds) // val_percentage  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(seed))
        train_loaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        val_loaders.append(DataLoader(ds_val, batch_size=batch_size, shuffle=False))
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    aux_loader = None
    if aux_set:
        aux_loader = DataLoader(aux_set, batch_size=batch_size, shuffle=False)
    return train_loaders, val_loaders, test_loader, aux_loader
