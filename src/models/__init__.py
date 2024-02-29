import torch.nn as nn
import torchvision

from models.mnist_net import emnist_resnet18
from src.models.cifar_resnets import get_resnet_model


def _get_imagenet_resnet(model_name: str = "resnet18", num_classes: int = 100, weights=None) -> nn.Module:
    match (model_name.lower()):
        case "resnet18":
            model = torchvision.models.resnet18(weights=weights)
        case "resnet50":
            model = torchvision.models.resnet50(weights=weights)
        case _:
            raise Exception(f"Model: '{model_name}' does not exists.")
    model.fc = nn.Linear(in_features=model.inplanes, out_features=num_classes, bias=True)
    return model


def get_model(model_name: str = "resnet32", dataset="cifar10", weights=None, device: str = "cpu") -> nn.Module:
    match (dataset.lower()):
        case "emnist":
            model = emnist_resnet18(num_classes=37)  # 10 digits, 36 letters + 1: "N/A"
        case "cifar10":
            model = get_resnet_model(model_name, num_classes=10)
        case "cifar100":
            model = get_resnet_model(model_name, num_classes=100)
        case "imagenet100":
            model = _get_imagenet_resnet(model_name=model_name, weights=weights, num_classes=100)
        case _:
            raise Exception(f"Dataset '{dataset}' does not exist.")
    model.to(device)
    return model
