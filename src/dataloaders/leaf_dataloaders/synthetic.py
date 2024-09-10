import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as t


class SyntheticDataset(Dataset):

    def __init__(self, leaf_ds_root: str):
        # Load dataset
        with open(os.path.join(leaf_ds_root, "synthetic", "data.json"), 'r') as f:
            data = json.load(f)
        all_x_values = []
        all_y_values = []
        # Itera attraverso le chiavi numeriche in "user_data"
        for key in data['user_data']:
            x_data = data['user_data'][key]['x']
            y_data = data['user_data'][key]['y']
            # Itera attraverso le sotto-chiavi di "x" e raccogli i valori
            for sub_x_data, sub_y_data in zip(x_data, y_data):
                all_x_values.append(sub_x_data)
                all_y_values.append(sub_y_data)
        # Set dataset
        self.data = list(zip(all_x_values, all_y_values))

    def __getitem__(self, index: int):
        data, label = self.data[index]
        data = np.expand_dims(np.array(data), axis=0)  # Make the list a 1-d image
        return data, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)


class SyntheticTransformedDataset(Dataset):

    def __init__(self, original_dataset: SyntheticDataset, transforms=None):
        self.original_dataset = original_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, label = self.original_dataset[idx]
        if self.transforms:
            image = self.transforms(image)
        return image, label


if __name__ == "__main__":
    ds = SyntheticDataset("../../../datasets/leaf/")
    item = ds.__getitem__(0)
    syn_transforms = t.Compose([
        t.ToTensor(),
        t.Normalize(0.7518, 1.4211)
    ])
    ds2 = SyntheticTransformedDataset(ds, transforms=syn_transforms)
    item2 = ds2.__getitem__(0)
    print("Item no transforms:", item)
    print("Item with transforms:", item2)
