from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100

DEFAULT_CIFAR_DATA = {
    "crop_size": 32,
    "pad_size": 4,
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
        "transforms": [
            transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]),
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        ],
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
        "transforms": [
            transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ]),
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        ],
    },
}


def get_cifar_10(path: str, transform=None) -> tuple[Dataset, Dataset]:
    """Load CIFAR-10 (training and test set)."""
    if transform is None:
        transform = DEFAULT_CIFAR_DATA["cifar10"]["transforms"]
    train_set = CIFAR10(path, train=True, download=True, transform=transform[0])
    test_set = CIFAR10(path, train=False, download=True, transform=transform[1])
    return train_set, test_set


def get_cifar_100(path: str, transform=None) -> tuple[Dataset, Dataset]:
    """Load CIFAR-100 (training and test set)."""
    if transform is None:
        transform = DEFAULT_CIFAR_DATA["cifar100"]["transforms"]
    train_set = CIFAR100(path, train=True, download=True, transform=transform[0])
    test_set = CIFAR100(path, train=False, download=True, transform=transform[1])
    return train_set, test_set
