import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def get_resnet_model(num_classes: int = 10, model_name: str = "resnet18", use_pretrain: bool = False,
                     use_groupnorm: bool = False, groupnorm_channels: int = 2) -> nn.Module:
    """Generates a ResNet18 model.

    Args:
        num_classes (int, optional): Number of classes. Defaults to 10.
        model_name (str, optional): Name of the model to use. Defaults to 'resnet18'.
        use_pretrain (bool): True if we use pretrained weights, False otherwise.
        use_groupnorm (bool, optional): True if we want to use GroupNorm instead of BatchNorm2d.
        groupnorm_channels (int, optional): Number of channels per group in GroupNorm.

    Returns:
        nn.Module: ResNet18 network.
    """
    if use_pretrain:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # Carica modello pre-addestrato
    else:
        model = resnet18(weights=None)  # Modello non pre-addestrato

    # Modifica il classificatore per adattarlo al numero di classi
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Sostituisci l'ultimo layer

    # Se richiesto, sostituisci BatchNorm con GroupNorm
    if use_groupnorm:
        model = _batch_norm_to_group_norm(model, groupnorm_channels)
    return model


def _batch_norm_to_group_norm(layer, groupnorm_channels: int = 2):
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
                if isinstance(sub_layer, nn.BatchNorm2d):
                    num_channels = sub_layer.num_features
                    # Make sure to have at least 1 channel and less or equal to num_channels
                    input_channels = max(min(groupnorm_channels, num_channels), 1)
                    # If the number of channels to use is -1, then use all the available channels
                    if groupnorm_channels == -1:
                        input_channels = num_channels
                    # first level of current layer or model contains a batch norm --> replacing.
                    layer._modules[name] = nn.GroupNorm(input_channels, num_channels)
            except AttributeError:
                # go deeper: set name to layer1, getattr will return layer1 --> call this func again
                name = name.split(".")[0]
                sub_layer = getattr(layer, name)
                sub_layer = _batch_norm_to_group_norm(sub_layer)
                layer.__setattr__(name=name, value=sub_layer)
    return layer
