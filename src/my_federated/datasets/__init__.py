from my_federated.datasets.custom.ds_medmnist import get_medmnist_num_classes


def get_dataset_classes(dataset_name: str):
    dataset_name = dataset_name.lower()
    match dataset_name:
        case "mnist":
            num_classes = 10
        case "fashionmnist":
            num_classes = 10
        case "cifar10":
            num_classes = 10
        case "cifar100":
            num_classes = 100
        case "caltech256":
            num_classes = 257  # 256 + 1 background
        case "eurosat":
            num_classes = 10
        case "imagenette":
            num_classes = 10
        case _:
            if dataset_name.startswith("medmnist_"):
                num_classes = get_medmnist_num_classes(dataset_name)
            else:
                raise Exception(f"Dataset '{dataset_name}' does not exist.")
    return num_classes
