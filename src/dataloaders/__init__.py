from src.dataloaders.cifar import get_cifar_10, get_cifar_100
from src.dataloaders.leaf_dataloaders import get_leaf_dataset
from src.dataloaders.mnist import get_emnist
from src.dataloaders.random_dataset import get_random_dataset
from src.dataloaders.utils import prepare_data, load_aux_dataset


def get_dataset(ds_root: str, ds_name: str,
                aux_seed: int = 0, generate_aux_set: bool = False):
    sample_shape = (3, 32, 32)
    samples = 10
    match ds_name:
        case "cifar10":
            train, test = get_cifar_10(ds_root)
        case "cifar100":
            train, test = get_cifar_100(ds_root)
        case "emnist":
            train, test = get_emnist(ds_root)
        case "leaf_celeba":
            sample_shape = (3, 84, 84)
            train, test = get_leaf_dataset(ds_root, ds_name)
        case "leaf_synthetic":
            sample_shape = (1, 10, 1)
            train, test = get_leaf_dataset(ds_root, ds_name)
        case _:
            assert False, f"Dataset '{ds_name}' not managed yet!"
    aux = None
    if generate_aux_set:
        aux = get_random_dataset(shape=sample_shape, number_samples=samples, aux_seed=aux_seed)
    return train, test, aux
