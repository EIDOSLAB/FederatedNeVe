from dataloaders.cifar import get_cifar_10, get_cifar_100
from dataloaders.utils import prepare_data


def get_dataset(ds_root: str, ds_name: str):
    assert ds_name in ["cifar10", "cifar100"]
    train, test = None, None
    match ds_name:
        case "cifar10":
            train, test = get_cifar_10(ds_root)
        case "cifar100":
            train, test = get_cifar_100(ds_root)
    return train, test
