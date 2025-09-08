import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset


class RandomDataset(Dataset):

    def __init__(self, img_shape: tuple[int, int, int] = (3, 32, 32), dataset_size: int = 10,
                 transform=None, seed: int = 0, labels_size: int = 1):
        self.img_shape = img_shape
        self.dataset_size = dataset_size
        self.transform = transform
        self.labels_size = labels_size
        self.seed = seed
        self.images = []
        self.generate()

    def __getitem__(self, index: int):
        img, label = self.images[index], 0

        if self.transform:
            img = self.transform(img)

        if self.labels_size > 1:
            label = np.array(self.labels_size * [label])
        if not isinstance(img, torch.Tensor):
            img = T.ToTensor()(img)
        return img, label

    def __len__(self):
        return len(self.images)

    def generate(self):

        transform = T.ToPILImage()
        self.images = []

        # Generate images
        for idx in range(self.dataset_size):
            print(f"Status: {idx + 1}/{self.dataset_size}")
            img = torch.randn(*self.img_shape)
            img = transform(img)
            self.images.append(img)


def get_random_dataset(shape: tuple[int, int, int] = (3, 32, 32), number_samples: int = 10, aux_seed: int = 0,
                       labels_size: int = 1):
    aux_dataset = RandomDataset(img_shape=shape, dataset_size=number_samples, seed=aux_seed, labels_size=labels_size)
    return aux_dataset
