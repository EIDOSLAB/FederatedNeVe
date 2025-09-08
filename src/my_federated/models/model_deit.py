import timm
import torch.nn as nn


def get_deit_model(num_classes: int = 10, model_name: str = "deit_tiny_patch16_224",
                   use_pretrain: bool = False) -> nn.Module:
    model = timm.create_model(model_name, pretrained=use_pretrain)
    num_features = model.head.in_features
    model.head = nn.Linear(num_features, num_classes)
    return model
