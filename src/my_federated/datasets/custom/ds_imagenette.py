import os

import torchvision.transforms as transforms
from torchvision.datasets import Imagenette

from my_federated.datasets.federated_dataset import FLDataset


class FLImagenetteDataset(FLDataset):
    def __init__(self, dataset_name: str, root_path: str, seed: int = 42):
        super().__init__(save_data_size=224, seed=seed, dataset_name=dataset_name, num_classes=10)

        # Download the data
        print("Preparing data...")
        os.makedirs(root_path, exist_ok=True)

        download = not os.path.exists(os.path.join(root_path, "imagenette2.tgz"))

        train_set = Imagenette(root_path, split="train", download=download, transform=self.get_save_transforms())
        test_set = Imagenette(root_path, split="val", download=False, transform=self.get_save_transforms())

        self.train_set = train_set
        self.test_set = test_set
        print("All data prepared.")


def get_imagenette(path: str, ds_name: str, seed: int) -> FLDataset:
    return FLImagenetteDataset(dataset_name=ds_name, root_path=path, seed=seed)


def get_imagenette_transforms(final_size: int = 224):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    return {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),  # Ridimensionamento casuale e crop
            transforms.RandomHorizontalFlip(),  # Flip orizzontale casuale
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
