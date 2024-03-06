from src.dataloaders.cifar import get_cifar_10, get_cifar_100
from src.dataloaders.mnist import get_emnist
from src.dataloaders.random import get_random_dataset
from src.dataloaders.utils import prepare_data


def get_dataset(ds_root: str, ds_name: str, seed: int = 0, generate_aux_set: bool = False):
    sample_shape = (3, 32, 32)
    samples = 10
    match ds_name:
        case "cifar10":
            train, test = get_cifar_10(ds_root)
        case "cifar100":
            train, test = get_cifar_100(ds_root)
        case "emnist":
            sample_shape = (3, 28, 28)
            train, test = get_emnist(ds_root)
        case _:
            assert False, f"Dataset '{ds_name}' not managed yet!"
    aux = None
    if generate_aux_set:
        aux = get_random_dataset(shape=sample_shape, number_samples=samples, seed=seed)
    return train, test, aux
