import torch.nn as nn

from my_federated.datasets import get_dataset_classes
from my_federated.models.model_deit import get_deit_model
from my_federated.models.model_efficientnet import get_efficientnet_model
from my_federated.models.model_resnet import get_resnet_model
from my_federated.models.tiny_vit import get_tiny_vit_model


def get_model(dataset="cifar10", model_name: str = "resnet18", device: str = "cuda",
              use_pretrain: bool = False,
              use_groupnorm=True, groupnorm_channels: int = 2) -> tuple[nn.Module, int]:
    num_classes = get_dataset_classes(dataset.lower())

    match model_name.lower():
        case "resnet18":
            model = get_resnet_model(num_classes=num_classes, model_name=model_name, use_pretrain=use_pretrain,
                                     use_groupnorm=use_groupnorm, groupnorm_channels=groupnorm_channels)
        case "efficientnet_b0":
            model = get_efficientnet_model(num_classes=num_classes, model_name=model_name, use_pretrain=use_pretrain,
                                           use_groupnorm=use_groupnorm, groupnorm_channels=groupnorm_channels)
        case "deit_tiny_patch16_224":
            model = get_deit_model(num_classes=num_classes, model_name=model_name, use_pretrain=use_pretrain)
        case "tiny_vit_5m_224":
            model = get_tiny_vit_model(num_classes=num_classes, model_name="tiny_vit_5m_224", use_pretrain=use_pretrain)
        case _:
            print(f"Model with name: {model_name} is not managed. A ResNet18 will be used instead.")
            model = get_resnet_model(num_classes=num_classes, model_name="resnet18", use_pretrain=use_pretrain,
                                     use_groupnorm=use_groupnorm, groupnorm_channels=groupnorm_channels)
    model.to(device)
    return model, num_classes
