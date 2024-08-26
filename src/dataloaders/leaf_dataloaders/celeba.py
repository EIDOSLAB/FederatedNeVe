import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CelebaDataset(Dataset):

    def __init__(self, leaf_ds_root: str):
        self.data = pd.read_csv(os.path.join(leaf_ds_root, "celeba", "labels.txt"))
        self.images_dir = os.path.join(leaf_ds_root, "celeba", "images")

    def __getitem__(self, index: int):
        if torch.is_tensor(index):
            index = index.item()

        img_name = os.path.join(self.images_dir, self.data.iloc[index, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.data.iloc[index, 1]  # Assuming labels are in the second column

        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)


class CelebaTransformedDataset(Dataset):

    def __init__(self, original_dataset: CelebaDataset, transforms=None):
        self.original_dataset = original_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, label = self.original_dataset[idx]
        if self.transforms:
            image = self.transforms(image)
        return image, label
