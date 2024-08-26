from src.models.federated_model import batch_norm_to_group_norm
from src.models.leaf_models.celeba import LeafCelebaModel
from src.models.leaf_models.synthetic import LeafSyntheticModel


def get_leaf_model(num_classes: int, ds_name: str,
                   use_groupnorm: bool = False, groupnorm_channels: int = 2,
                   leaf_input_dim: int = 10):
    match ds_name.lower():
        case "leaf_celeba":
            model = LeafCelebaModel(num_classes=num_classes)
        case "leaf_synthetic":
            model = LeafSyntheticModel(num_classes=num_classes, input_dim=leaf_input_dim)
        case _:
            assert False, f"Leaf model for dataset'{ds_name}' not managed yet!"
    if use_groupnorm:
        model = batch_norm_to_group_norm(model, groupnorm_channels)
    return model
