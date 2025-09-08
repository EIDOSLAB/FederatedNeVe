import os

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100

from my_federated.datasets.federated_dataset import FLDataset


class FLCIFARDataset(FLDataset):
    def __init__(self, dataset_name: str, root_path: str, seed: int = 42):
        super().__init__(save_data_size=32, seed=seed, dataset_name=dataset_name,
                         num_classes=10 if dataset_name == 'cifar10' else 100)

        # Download the data
        print("Preparing data...")
        os.makedirs(root_path, exist_ok=True)

        if self._dataset_name == 'cifar10':
            train_set = CIFAR10(root_path, train=True, download=True, transform=self.get_save_transforms())
            test_set = CIFAR10(root_path, train=False, download=True, transform=self.get_save_transforms())
        else:
            train_set = CIFAR100(root_path, train=True, download=True, transform=self.get_save_transforms())
            test_set = CIFAR100(root_path, train=False, download=True, transform=self.get_save_transforms())

        self.train_set = train_set
        self.test_set = test_set
        print("All data prepared.")


def get_cifar(path: str, ds_name: str, seed: int) -> FLDataset:
    return FLCIFARDataset(dataset_name=ds_name, root_path=path, seed=seed)


def get_cifar_transforms(final_size: int = 32):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    return {
        "train": transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((final_size, final_size)),
            transforms.RandAugment(),  # Applica augmentazioni casuali
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        "test": transforms.Compose([
            transforms.Resize((final_size, final_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }
