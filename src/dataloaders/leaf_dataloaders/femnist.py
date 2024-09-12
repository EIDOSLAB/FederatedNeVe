import io
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class FEmnistDataset(Dataset):

    def __init__(self, leaf_ds_root: str, split_type: str = "writer"):
        femnist_file = os.path.join(leaf_ds_root, "femnist", "data.pkl")
        os.makedirs(os.path.join(leaf_ds_root, "femnist"), exist_ok=True)
        if os.path.exists(femnist_file):
            self.data = pd.read_pickle(femnist_file)
        else:
            print("Downloading femnist dataset from huggingface...")
            self.data = pd.read_parquet("hf://datasets/flwrlabs/femnist/data/train-00000-of-00001.parquet")
            print("Saving femnist dataset into disk.")
            self.data.to_pickle(femnist_file)
            print("Done.")
        if split_type == "writer":
            # (writer_id__string)
            self.column_split = "writer_id"
        else:
            # (class_id__int)
            self.column_split = "character"

    def __getitem__(self, index: int):
        image, label = self._get_image(index, only_writer=False)
        return image, torch.tensor(label, dtype=torch.long)

    def get_writer_id(self, index: int):
        writer_id = self._get_image(index, only_writer=True)
        return writer_id

    def __len__(self):
        return len(self.data)

    def _get_image(self, index: int, only_writer: bool = False):
        data = self.data.iloc[index]
        if only_writer:
            return data["writer_id"]
        data_img = data["image"]
        # Convertire la stringa per essere compatibile con ast.literal_eval
        image = Image.open(io.BytesIO(data_img['bytes']))
        if image.mode == "L":
            image = image.convert("RGB")
        return image, data["character"]


if __name__ == "__main__":
    ds = FEmnistDataset("../../../datasets/leaf/")
    img, lbl = ds.__getitem__(0)
    print(ds)
