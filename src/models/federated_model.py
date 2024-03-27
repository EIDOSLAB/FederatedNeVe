import torch
from torch.nn import Module
from torchvision.models import resnet18, ResNet


def get_resnet_model(num_classes: int = 10, use_groupnorm: bool = False) -> Module:
    """Generates ResNet18 model using GroupNormalization rather than
    BatchNormalization. Two groups are used.

    Args:
        num_classes (int, optional): Number of classes. Defaults to 10.
        use_groupnorm (bool): True if we want to use GroupNorm rather than BatchNorm2d

    Returns:
        Module: ResNet18 network.
    """
    model: ResNet = resnet18(num_classes=num_classes)
    if use_groupnorm:
        model = batch_norm_to_group_norm(model)
    return model


def batch_norm_to_group_norm(layer):
    """Iterates over a whole model (or layer of a model) and replaces every batch norm 2D with a group norm

    Args:
        layer: model or one layer of a model like resnet34.layer1 or Sequential(), ...
    """
    for name, module in layer.named_modules():
        if name:
            try:
                # name might be something like: model.layer1.sequential.0.conv1 --> this wont work. Except this case
                sub_layer = getattr(layer, name)
                if isinstance(sub_layer, torch.nn.BatchNorm2d):
                    num_channels = sub_layer.num_features
                    # first level of current layer or model contains a batch norm --> replacing.
                    layer._modules[name] = torch.nn.GroupNorm(2, num_channels)
            except AttributeError:
                # go deeper: set name to layer1, getattr will return layer1 --> call this func again
                name = name.split(".")[0]
                sub_layer = getattr(layer, name)
                sub_layer = batch_norm_to_group_norm(sub_layer)
                layer.__setattr__(name=name, value=sub_layer)
    return layer
