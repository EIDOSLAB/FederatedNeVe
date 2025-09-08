import abc

import torch
from torch.utils.data import random_split
from torchvision import transforms


class FLDataset(abc.ABC):

    def __init__(self, save_data_size: int, seed: int, dataset_name: str, num_classes: int):
        self._num_classes = num_classes
        self._dataset_name = dataset_name.lower()
        self._dataset_task = "multi-class"
        self._save_data_size = save_data_size
        self.seed = seed
        self.train_set = None
        self.test_set = None

    def get_save_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize((self._save_data_size, self._save_data_size)),
                transforms.ToTensor()
            ]
        )

    def get_num_classes(self):
        return self._num_classes

    def get_task_name(self):
        return self._dataset_task

    def _split_dataset(self, dataset, splits: list[int]):
        # Suddividere il dataset con random_split
        generator = torch.Generator().manual_seed(self.seed)
        return random_split(range(len(dataset)), splits, generator=generator)
