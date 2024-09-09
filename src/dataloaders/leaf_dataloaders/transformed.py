from torch.utils.data import Dataset


class LeafTransformedDataset(Dataset):

    def __init__(self, original_dataset: Dataset, transforms=None):
        self.original_dataset = original_dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, label = self.original_dataset[idx]
        if self.transforms:
            image = self.transforms(image)
        return image, label
