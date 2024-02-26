from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import EMNIST

_DEFAULT_MNIST_DATA = {
    "emnist": {
        "mean": (0.1307,),
        "std": (0.3081,),
        "transforms": [
            # Train
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]),
            # Validation/Test
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        ],
    },
}
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


def get_emnist(path: str, split: str = "letter", transform=None) -> tuple[Dataset, Dataset]:
    """Load EMNIST (training and test set)."""
    if transform is None:
        transform = _DEFAULT_MNIST_DATA["emnist"]["transforms"]
    train_set = EMNIST(path, split=split, train=True, download=True, transform=transform[0])
    test_set = EMNIST(path, split=split, train=False, download=True, transform=transform[1])
    return train_set, test_set
