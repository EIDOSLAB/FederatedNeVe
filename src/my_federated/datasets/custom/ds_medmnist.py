import os

import medmnist
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset

from my_federated.datasets.federated_dataset import FLDataset


class FLMedmnistDataset(FLDataset):
    def __init__(self, dataset_name: str, root_path: str, save_data_size: int = 224, seed: int = 42,
                 medmnist_size: int = 224):
        super().__init__(save_data_size=save_data_size, seed=seed, dataset_name=dataset_name, num_classes=10)
        assert self._dataset_name.startswith("medmnist")
        self._dataset_name = self._dataset_name.split("_")[-1]
        info = medmnist.INFO[self._dataset_name]
        DataClass = getattr(medmnist, info["python_class"])
        self._medmnist_size = medmnist_size
        assert self._medmnist_size in DataClass.available_sizes
        self._num_classes = get_medmnist_num_classes(self._dataset_name)
        self._dataset_task = info["task"]
        os.makedirs(root_path, exist_ok=True)

        # Download the data
        print("Preparing training data...")
        train_set = DataClass(root=root_path, split="train", as_rgb=True, download=True, size=self._medmnist_size,
                              transform=self.get_save_transforms())
        valid_set = DataClass(root=root_path, split="val", as_rgb=True, download=True, size=self._medmnist_size,
                              transform=self.get_save_transforms())
        self.train_set = ConcatDataset([train_set, valid_set])

        print("Preparing test data...")
        self.test_set = DataClass(root=root_path, split="test", as_rgb=True, download=True,
                                  size=self._medmnist_size,
                                  transform=self.get_save_transforms())
        print("All data prepared.")


def download_all_medmnist(path, size: int = 224):
    for ds_name in medmnist.INFO.keys():
        info = medmnist.INFO[ds_name]
        DataClass = getattr(medmnist, info["python_class"])
        assert size in DataClass.available_sizes
        # load the data
        _ = DataClass(root=path, split="train", as_rgb=True, download=True, size=size)
        _ = DataClass(root=path, split="val", as_rgb=True, download=True, size=size)
        _ = DataClass(root=path, split="test", as_rgb=True, download=True, size=size)


def get_medmnist(path: str, ds_name: str, seed: int, medmnist_size: int = 224) -> FLDataset:
    return FLMedmnistDataset(dataset_name=ds_name, root_path=path, medmnist_size=medmnist_size, seed=seed)


def get_medmnist_transforms(final_size: int = 224):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    return {
        "train": transforms.Compose([
            transforms.RandomRotation(5),  # Rotazione casuale
            transforms.Resize((final_size, final_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        "test": transforms.Compose([
            transforms.Resize((final_size, final_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }


def get_medmnist_num_classes(ds_name: str):
    if ds_name.startswith("medmnist_"):
        ds_name = ds_name.split("_")[-1]
    info = medmnist.INFO[ds_name]
    n_classes = len(info["label"])
    return n_classes
