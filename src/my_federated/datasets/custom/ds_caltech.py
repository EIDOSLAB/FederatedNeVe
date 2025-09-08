import os

import torch
from torchvision import transforms
from torchvision.datasets import Caltech256

from my_federated.datasets.federated_dataset import FLDataset


class FLCaltech256Dataset(FLDataset):
    def __init__(self, dataset_name: str, root_path: str, seed: int = 42):
        super().__init__(save_data_size=256, seed=seed, dataset_name=dataset_name, num_classes=257)

        # Download the data
        print("Preparing data...")
        os.makedirs(root_path, exist_ok=True)
        dataset = Caltech256(root=root_path, download=True, transform=self.get_save_transforms())

        # Definizione delle dimensioni dello split
        train_size = int(0.8 * len(dataset))  # 80% train
        test_size = len(dataset) - train_size  # 20% test

        train_indices, test_indices = self._split_dataset(dataset, [train_size, test_size])

        # Creiamo i sotto-dataset applicando le giuste trasformazioni
        self.train_set = torch.utils.data.Subset(dataset, train_indices)
        self.test_set = torch.utils.data.Subset(dataset, test_indices)
        print("All data prepared.")


def get_caltech(path: str, ds_name: str, seed: int) -> FLDataset:
    if ds_name == "caltech256":
        return FLCaltech256Dataset(dataset_name=ds_name, root_path=path, seed=seed)
    raise AssertionError(f"Caltech with name: '{ds_name}' is not supported.")


def get_caltech_transforms(final_size: int = 224):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return {
        "train": transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop((final_size, final_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        "test": transforms.Compose([
            transforms.Resize((final_size, final_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }
