import torch.nn as nn

from src.models.federated_model import get_resnet_model


def get_model(dataset="cifar10", device: str = "cpu", use_groupnorm=True) -> nn.Module:
    match (dataset.lower()):
        case "emnist":
            num_classes = 37  # 10 digits, 26 letters + 1: "N/A"
        case "cifar10":
            num_classes = 10
        case "cifar100":
            num_classes = 100
        case _:
            raise Exception(f"Dataset '{dataset}' does not exist.")
    model = get_resnet_model(num_classes=num_classes, use_groupnorm=use_groupnorm)
    model.to(device)
    return model
