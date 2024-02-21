import random

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset


class RandomDataset(Dataset):

    def __init__(self, img_shape: tuple[int, int, int] = (3, 32, 32), dataset_size: int = 10,
                 transform=None, seed: int = 0):
        self.img_shape = img_shape
        self.dataset_size = dataset_size
        self.transform = transform
        self.seed = seed
        self.images = []
        self.generate()

    def __getitem__(self, index: int):
        img = self.images[index]

        if self.transform:
            img = self.transform(img)
        img = T.Resize((self.img_shape[1], self.img_shape[2]))(img)  # Make sure the image is of the correct size

        return T.ToTensor()(img), 0

    def __len__(self):
        return len(self.images)

    def generate(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        transform = T.ToPILImage()
        self.images = []

        # Generate images
        for idx in range(self.dataset_size):
            print(f"Status: {idx + 1}/{self.dataset_size}")
            img = torch.randn(*self.img_shape)
            img = transform(img)
            self.images.append(img)


def get_random_dataset(shape: tuple[int, int, int] = (3, 32, 32), number_samples: int = 10, seed: int = 0):
    aux_dataset = RandomDataset(img_shape=shape, dataset_size=number_samples, seed=seed)
    return aux_dataset
