import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from torchvision.datasets import EMNIST

_DEFAULT_MNIST_DATA = {
    "emnist": {
        "mean": (0.1307, 0.1307, 0.1307),
        "std": (0.3081, 0.3081, 0.3081),
        "transforms": [
            # Train
            transforms.Compose([
                transforms.RandomCrop(24),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
            ]),
            # Validation/Test
            transforms.Compose([
                transforms.CenterCrop(24),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081))
            ])
        ],
    },
}


def get_emnist(path: str, transform=None) -> tuple[Dataset, Dataset]:
    """Load EMNIST (training and test set)."""
    if transform is None:
        transform = _DEFAULT_MNIST_DATA["emnist"]["transforms"]
    train_digits_set = EMNIST(path, split="digits", train=True, download=True, transform=transform[0])
    test_digits_set = EMNIST(path, split="digits", train=False, download=True, transform=transform[1])
    train_letters_set = EMNIST(path, split="letters", train=True, download=True, transform=transform[0])
    test_letters_set = EMNIST(path, split="letters", train=False, download=True, transform=transform[1])
    # Modifica le etichette del dataset "letters" per farle partire da 10 (in digits ci sono 10 classi)
    offset = 10
    train_letters_set.targets += offset
    test_letters_set.targets += offset
    # Combina i dataset
    emnist_combined_train = ConcatDataset([train_digits_set, train_letters_set])
    emnist_combined_test = ConcatDataset([test_digits_set, test_letters_set])

    train_size = int(0.1 * len(emnist_combined_train))
    test_size = len(emnist_combined_train) - train_size
    emnist_combined_train, _ = torch.utils.data.random_split(emnist_combined_train, [train_size, test_size])

    return emnist_combined_train, emnist_combined_test
