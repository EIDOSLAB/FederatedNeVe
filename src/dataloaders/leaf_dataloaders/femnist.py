import io
import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class FEmnistDataset(Dataset):

    def __init__(self, leaf_ds_root: str, split_type: str = "writer", late_init: bool = False):
        femnist_file = os.path.join(leaf_ds_root, "femnist", "data.pkl")
        os.makedirs(os.path.join(leaf_ds_root, "femnist"), exist_ok=True)
        if late_init:
            self.data: pd.DataFrame | None = None
        else:
            if os.path.exists(femnist_file):
                self.data: pd.DataFrame = pd.read_pickle(femnist_file)
            else:
                print("Downloading femnist dataset from huggingface...")
                femnist_link = "hf://datasets/flwrlabs/femnist/data/train-00000-of-00001.parquet"
                self.data: pd.DataFrame = pd.read_parquet(femnist_link)
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
        image, label = self._get_image(index)
        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def _get_image(self, index: int):
        data = self.data.iloc[index]
        data_img = data["image"]
        # Convertire la stringa per essere compatibile con ast.literal_eval
        image = Image.open(io.BytesIO(data_img['bytes']))
        if image.mode == "L":
            image = image.convert("RGB")
        return image, data["character"]

    def late_init(self, data: pd.DataFrame):
        self.data = data

    def get_unique_writers(self):
        return self.data[self.column_split].unique().tolist()

    def get_partition_by_filter(self, filter_val: str | list[str]):
        if isinstance(filter_val, str):
            filter_val = [filter_val]
        return self.data[self.data[self.column_split].isin(filter_val)]


if __name__ == "__main__":
    ds = FEmnistDataset("../../../datasets/leaf/")
    img, lbl = ds.__getitem__(0)
    print(ds)
