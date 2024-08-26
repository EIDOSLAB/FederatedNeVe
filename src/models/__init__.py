import torch.nn as nn

from src.models.leaf_models import get_leaf_model
from src.models.federated_model import get_resnet_model, get_efficientnet_model


def get_model(dataset="cifar10", model_name: str = "resnet18", device: str = "cuda",
              use_pretrain: bool = False,
              use_groupnorm=True, groupnorm_channels: int = 2, leaf_input_dim: int = 10) -> tuple[nn.Module, int]:
    match (dataset.lower()):
        case "emnist":
            num_classes = 37  # 10 digits, 26 letters + 1: "N/A"
        case "cifar10":
            num_classes = 10
        case "cifar100":
            num_classes = 100
        case "celeba":
            num_classes = 2
        case _:
            raise Exception(f"Dataset '{dataset}' does not exist.")
    if "efficientnet" in model_name.lower():
        model = get_efficientnet_model(num_classes=num_classes, model_name=model_name, use_pretrain=use_pretrain,
                                       use_groupnorm=use_groupnorm, groupnorm_channels=groupnorm_channels)
    elif "resnet" in model_name.lower():
        model = get_resnet_model(num_classes=num_classes, model_name=model_name, use_pretrain=use_pretrain,
                                 use_groupnorm=use_groupnorm, groupnorm_channels=groupnorm_channels)
    elif "leaf" in model_name.lower():
        model = get_leaf_model(num_classes=num_classes, ds_name=dataset.lower(),
                               use_groupnorm=use_groupnorm, groupnorm_channels=groupnorm_channels,
                               leaf_input_dim=leaf_input_dim)
    else:
        print(f"Model with name: {model_name} is not managed. A ResNet18 will be used instead.")
        model = get_resnet_model(num_classes=num_classes, model_name="resnet18", use_pretrain=use_pretrain,
                                 use_groupnorm=use_groupnorm, groupnorm_channels=groupnorm_channels)
    model.to(device)
    return model, num_classes
