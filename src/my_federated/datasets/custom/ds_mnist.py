import os

import torchvision.transforms as transforms
from torchvision.datasets import MNIST, FashionMNIST

from my_federated.datasets.federated_dataset import FLDataset


class FLMnistDataset(FLDataset):
    def __init__(self, dataset_name: str, root_path: str, seed: int = 42):
        super().__init__(save_data_size=28, seed=seed, dataset_name=dataset_name, num_classes=10)

        # Download the data
        print("Preparing data...")
        os.makedirs(root_path, exist_ok=True)

        if self._dataset_name == 'mnist':
            train_set = MNIST(root_path, train=True, download=True, transform=self.get_save_transforms())
            test_set = MNIST(root_path, train=False, download=True, transform=self.get_save_transforms())
        else:
            train_set = FashionMNIST(root_path, train=True, download=True, transform=self.get_save_transforms())
            test_set = FashionMNIST(root_path, train=False, download=True, transform=self.get_save_transforms())

        self.train_set = train_set
        self.test_set = test_set
        print("All data prepared.")


def get_mnist(path: str, ds_name: str, seed: int) -> FLDataset:
    return FLMnistDataset(dataset_name=ds_name, root_path=path, seed=seed)


def get_mnist_transforms(ds_name: str, final_size: int = 32):
    if ds_name == "mnist":
        mean = (0.1307, 0.1307, 0.1307)
        std = (0.3081, 0.3081, 0.3081)
    else:
        mean = (0.2860, 0.2860, 0.2860)
        std = (0.3530, 0.3530, 0.3530)

    return {
        "train": transforms.Compose([
            transforms.RandomRotation(10),  # Rotazione casuale
            transforms.RandomAffine(0, shear=10),  # Affine (scorrimento, rotazione)
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
