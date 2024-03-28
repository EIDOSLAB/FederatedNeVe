import os
import random

import numpy as np
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR

from src.NeVe.scheduler import ReduceLROnLocalPlateau, NeVeScheduler


def set_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def get_optimizer(model: nn.Module, opt_name: str = "sgd", starting_lr: float = 0.1, weight_decay: float = 5e-4,
                  momentum: float = 0.9) -> torch.optim.Optimizer:
    print(f"Initialize optimizer: {opt_name}")
    match (opt_name.lower()):
        case "sgd":
            optim = SGD(model.parameters(), lr=starting_lr, weight_decay=weight_decay, momentum=momentum)
        case "adam":
            optim = Adam(model.parameters(), lr=starting_lr, weight_decay=weight_decay)
        case _:
            raise Exception(f"Optimizer '{opt_name}' not defined.")
    return optim


def get_scheduler(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler_name: str = "neve",
                  dataset: str = "cifar10"):
    if scheduler_name == "neve":
        return NeVeScheduler(model, lr_scheduler=ReduceLROnLocalPlateau(optimizer))

    match dataset.lower():
        case "emnist":
            milestones = [30, 60]
        case _:
            milestones = [100, 150]
    return MultiStepLR(optimizer, milestones=milestones)
