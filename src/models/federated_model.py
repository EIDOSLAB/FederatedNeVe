from torch.nn import Module, GroupNorm
from torchvision.models import resnet18, ResNet


def get_resnet_model(num_classes: int = 10) -> Module:
    """Generates ResNet18 model using GroupNormalization rather than
    BatchNormalization. Two groups are used.

    Args:
        num_classes (int, optional): Number of classes. Defaults to 10.

    Returns:
        Module: ResNet18 network.
    """
    model: ResNet = resnet18(
        norm_layer=lambda x: GroupNorm(2, x), num_classes=num_classes
    )
    return model
