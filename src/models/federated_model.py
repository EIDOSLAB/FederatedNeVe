import torch
from torch.nn import Module
from torchvision.models import EfficientNet, efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import ResNet, resnet18, ResNet18_Weights


def get_resnet_model(num_classes: int = 10, model_name: str = "resnet18", use_pretrain: bool = False,
                     use_groupnorm: bool = False, groupnorm_channels: int = 2) -> Module:
    """Generates ResNet18 model

    Args:
        num_classes (int, optional): Number of classes. Defaults to 10.
        model_name (str, optional): Name of the model to use. Defaults to resnet18
        use_pretrain (bool): True if we use pretrain weights, False otherwise
        use_groupnorm (bool, optional): True if we want to use GroupNorm rather than BatchNorm2d
        groupnorm_channels (int, optional): Number of channels used to split data-channels from the GroupNorm

    Returns:
        Module: ResNet18 network.
    """
    if use_pretrain:
        model: ResNet = resnet18(num_classes=num_classes, weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        model: ResNet = resnet18(num_classes=num_classes)
    if use_groupnorm:
        model = batch_norm_to_group_norm(model, groupnorm_channels)
    return model


def get_efficientnet_model(num_classes: int = 10, model_name: str = "efficientnet_b0", use_pretrain: bool = False,
                           use_groupnorm: bool = False, groupnorm_channels: int = 2) -> Module:
    """Generates Efficientnet model

    Args:
        num_classes (int, optional): Number of classes. Defaults to 10.
        model_name (str, optional): Name of the model to use. Defaults to efficientnet_b0
        use_pretrain (bool): True if we use pretrain weights, False otherwise
        use_groupnorm (bool, optional): True if we want to use GroupNorm rather than BatchNorm2d
        groupnorm_channels (int, optional): Number of channels used to split data-channels from the GroupNorm

    Returns:
        Module: Efficientnet network.
    """
    if use_pretrain:
        model: EfficientNet = efficientnet_b0(num_classes=num_classes, weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    else:
        model: EfficientNet = efficientnet_b0(num_classes=num_classes)
    if use_groupnorm:
        model = batch_norm_to_group_norm(model, groupnorm_channels)
    return model


def batch_norm_to_group_norm(layer, groupnorm_channels: int = 2):
    """Iterates over a whole model (or layer of a model) and replaces every batch norm 2D with a group norm

    Args:
        layer: model or one layer of a model like resnet34.layer1 or Sequential(), ...
        groupnorm_channels: number of channels to use in the GroupNorm
    """
    for name, module in layer.named_modules():
        if name:
            try:
                # name might be something like: model.layer1.sequential.0.conv1 --> this wont work. Except this case
                sub_layer = getattr(layer, name)
                if isinstance(sub_layer, torch.nn.BatchNorm2d):
                    num_channels = sub_layer.num_features
                    # Make sure to have at least 1 channel and less or equal to num_channels
                    input_channels = max(min(groupnorm_channels, num_channels), 1)
                    # If the number of channels to use is -1, then use all the available channels
                    if groupnorm_channels == -1:
                        input_channels = num_channels
                    # first level of current layer or model contains a batch norm --> replacing.
                    layer._modules[name] = torch.nn.GroupNorm(input_channels, num_channels)
            except AttributeError:
                # go deeper: set name to layer1, getattr will return layer1 --> call this func again
                name = name.split(".")[0]
                sub_layer = getattr(layer, name)
                sub_layer = batch_norm_to_group_norm(sub_layer)
                layer.__setattr__(name=name, value=sub_layer)
    return layer
