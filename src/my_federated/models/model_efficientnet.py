import torch.nn as nn
from torchvision.models.efficientnet import efficientnet_b0, EfficientNet_B0_Weights


def get_efficientnet_model(num_classes: int = 10, model_name: str = "efficientnet_b0", use_pretrain: bool = False,
                           use_groupnorm: bool = False, groupnorm_channels: int = 2) -> nn.Module:
    """Generates an EfficientNet model.

    Args:
        num_classes (int, optional): Number of classes. Defaults to 10.
        model_name (str, optional): Name of the model to use. Defaults to 'efficientnet_b0'.
        use_pretrain (bool): True if we use pre-trained weights, False otherwise.
        use_groupnorm (bool, optional): True if we want to use GroupNorm instead of BatchNorm2d.
        groupnorm_channels (int, optional): Number of channels per group in GroupNorm.

    Returns:
        nn.Module: EfficientNet network.
    """
    if use_pretrain:
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    else:
        model = efficientnet_b0(weights=None)

    # Modifica il classifier per adattarlo al numero di classi desiderato
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)  # Sostituisci solo l'ultimo layer

    # Se richiesto, sostituisci BatchNorm con GroupNorm
    if use_groupnorm:
        model = _batch_norm_to_group_norm(model, groupnorm_channels)

    return model


def _batch_norm_to_group_norm(model, groupnorm_channels: int = 2):
    """
    Sostituisce tutti i layer BatchNorm2d con GroupNorm nel modello.

    Args:
        model (nn.Module): Il modello di rete neurale.
        groupnorm_channels: number of channels to use in the GroupNorm

    Returns:
        nn.Module: Il modello con GroupNorm al posto di BatchNorm2d.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            # Estraiamo i parametri da BatchNorm
            num_features = module.num_features
            # Creiamo un GroupNorm con il numero di gruppi specificato
            groupnorm = nn.GroupNorm(groupnorm_channels, num_features)

            # Sostituire il BatchNorm con GroupNorm
            setattr(model, name, groupnorm)
        else:
            # Chiamare ricorsivamente su moduli figli
            _batch_norm_to_group_norm(module, groupnorm_channels)

    return model
