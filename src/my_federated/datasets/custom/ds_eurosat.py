import os

import torch
import torchvision.transforms as transforms
from torchvision.datasets import EuroSAT

from my_federated.datasets.federated_dataset import FLDataset


class FLEurosatDataset(FLDataset):
    def __init__(self, dataset_name: str, root_path: str, seed: int = 42):
        super().__init__(save_data_size=64, seed=seed, dataset_name=dataset_name, num_classes=10)

        # Download the data
        print("Preparing data...")
        os.makedirs(root_path, exist_ok=True)

        # Caricamento del dataset due volte (una per train, una per test)
        dataset = EuroSAT(root=root_path, download=True, transform=self.get_save_transforms())

        # Definizione delle dimensioni dello split
        train_size = int(0.8 * len(dataset))  # 80% train
        test_size = len(dataset) - train_size  # 20% test
        train_indices, test_indices = self._split_dataset(dataset, [train_size, test_size])

        # Creiamo i sotto-dataset applicando le giuste trasformazioni
        self.train_set = torch.utils.data.Subset(dataset, train_indices)
        self.test_set = torch.utils.data.Subset(dataset, test_indices)
        print("All data prepared.")


def get_eurosat(path: str, ds_name: str, seed: int) -> FLDataset:
    return FLEurosatDataset(dataset_name=ds_name, root_path=path, seed=seed)


def get_eurosat_transforms(final_size: int = 224):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    return {
        "train": transforms.Compose([
            transforms.RandomHorizontalFlip(),  # Flip orizzontale casuale
            transforms.RandomVerticalFlip(),  # Flip verticale casuale
            transforms.RandomRotation(10),  # Rotazione casuale
            transforms.RandomCrop(64, padding=4),  # Crop casuale con padding
            transforms.Resize((final_size, final_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        "test": transforms.Compose([
            transforms.Resize((final_size, final_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }
