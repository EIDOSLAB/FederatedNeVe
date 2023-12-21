import torchvision
from torch import nn


# ResNet V1.5 -> https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch
def resnet50(num_classes=1000, weights=None):
    return _build_model("resnet50", num_classes, weights)


def resnet18(num_classes=1000, weights=None):
    return _build_model("resnet18", num_classes, weights)


def _build_model(model_name, num_classes, weights):
    assert model_name in ["resnet50", "resnet18"]
    if model_name == "resnet50":
        resnet_model = torchvision.models.resnet50(weights=weights)
    else:
        resnet_model = torchvision.models.resnet18(weights=weights)
    resnet_model.fc = nn.Linear(in_features=resnet_model.inplanes, out_features=num_classes, bias=True)
    return resnet_model
