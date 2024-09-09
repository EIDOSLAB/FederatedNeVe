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
        img_name = os.path.join(self.images_dir, self.data.iloc[index, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.data.iloc[index, 1]  # Assuming labels are in the second column

        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)
