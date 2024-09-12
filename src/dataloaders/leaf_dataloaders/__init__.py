import os.path

import torch
from torch.utils.data import random_split
from torchvision import transforms

from src.dataloaders.leaf_dataloaders.celeba import CelebaDataset
from src.dataloaders.leaf_dataloaders.femnist import FEmnistDataset
from src.dataloaders.leaf_dataloaders.synthetic import SyntheticDataset, SyntheticTransformedDataset
from src.dataloaders.leaf_dataloaders.transformed import LeafTransformedDataset

DATA_TRANSFORMS = {
    "femnist": {
        "mean": (0.9627, 0.9627, 0.9627),
        "std": (0.1550, 0.1550, 0.1550),
        "transforms": [
            # Train
            transforms.Compose([
                transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=10),
                transforms.CenterCrop((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.9627, 0.9627, 0.9627), (0.1550, 0.1550, 0.1550))
            ]),
            # Validation/Test
            transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.9627, 0.9627, 0.9627), (0.1550, 0.1550, 0.1550))
            ])
        ],
    },
    "celeba": {
        "mean": (0.5061, 0.4254, 0.3828),
        "std": (0.2659, 0.2452, 0.2413),
        "transforms": [
            # Train
            transforms.Compose([
                transforms.RandomResizedCrop((84, 84), scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5061, 0.4254, 0.3828), (0.2659, 0.2452, 0.2413))
            ]),
            # Validation/Test
            transforms.Compose([
                transforms.Resize((84, 84)),
                transforms.ToTensor(),
                transforms.Normalize((0.5061, 0.4254, 0.3828), (0.2659, 0.2452, 0.2413))
            ])
        ],
    },
    "synthetic": {
        "mean": 0.7518,
        "std": 1.4211,
        "transforms": [
            # Train
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(0.7518, 1.4211)
            ]),
            # Validation/Test
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(0.7518, 1.4211)
            ])
        ],
    },
}


def get_leaf_dataset(root: str, dataset_name: str, seed: int = 0, train_size: float = 0.7):
    leaf_root = os.path.join(root, "leaf")
    os.makedirs(leaf_root, exist_ok=True)
    match dataset_name:
        case "leaf_femnist_bywriter":
            train, test = get_femnist(leaf_root, seed, train_size=train_size, split_type="writer")
        case "leaf_femnist_byclass":
            train, test = get_femnist(leaf_root, seed, train_size=train_size, split_type="class")
        case "leaf_celeba":
            train, test = get_celeba(leaf_root, seed, train_size=train_size)
        case "leaf_synthetic":
            train, test = get_synthetic(leaf_root, seed, train_size=train_size)
        case _:
            assert False, f"Leaf Dataset '{dataset_name}' not managed yet!"
    return train, test


def get_femnist(leaf_ds_root: str, seed: int = 0, train_size: float = 0.7, split_type: str = "writer"):
    femnist_dataset = FEmnistDataset(leaf_ds_root, split_type)
    train_ds, test_ds = split_dataset(femnist_dataset, seed, train_size=train_size)
    # Applicare le trasformazioni ai subset
    train_ds = LeafTransformedDataset(train_ds, DATA_TRANSFORMS["femnist"]["transforms"][0])
    test_ds = LeafTransformedDataset(test_ds, DATA_TRANSFORMS["femnist"]["transforms"][1])
    return train_ds, test_ds


def get_celeba(leaf_ds_root: str, seed: int = 0, train_size: float = 0.7):
    celeba_dataset = CelebaDataset(leaf_ds_root)
    train_ds, test_ds = split_dataset(celeba_dataset, seed, train_size=train_size)
    # Applicare le trasformazioni ai subset
    train_ds = LeafTransformedDataset(train_ds, DATA_TRANSFORMS["celeba"]["transforms"][0])
    test_ds = LeafTransformedDataset(test_ds, DATA_TRANSFORMS["celeba"]["transforms"][1])
    return train_ds, test_ds


def get_synthetic(leaf_ds_root: str, seed: int = 0, train_size: float = 0.7):
    synthetic_dataset = SyntheticDataset(leaf_ds_root)
    train_ds, test_ds = split_dataset(synthetic_dataset, seed, train_size=train_size)
    # Applicare le trasformazioni ai subset
    train_ds = SyntheticTransformedDataset(train_ds, DATA_TRANSFORMS["synthetic"]["transforms"][0])
    test_ds = SyntheticTransformedDataset(test_ds, DATA_TRANSFORMS["synthetic"]["transforms"][1])
    return train_ds, test_ds


def split_dataset(dataset, seed: int, train_size: float = 0.7):
    train_size = int(train_size * len(dataset))
    test_size = len(dataset) - train_size

    # Crea un generatore casuale con un seed specifico
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Esegui lo split
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    return train_dataset, test_dataset
