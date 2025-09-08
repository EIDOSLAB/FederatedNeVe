import os
import random

import numpy as np
import torch
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import MultiStepLR, ConstantLR


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
        case "adamw":
            optim = AdamW(model.parameters(), lr=starting_lr, weight_decay=weight_decay)
        case _:
            raise Exception(f"Optimizer '{opt_name}' not defined.")
    return optim


def get_scheduler(optimizer: torch.optim.Optimizer, scheduler_name: str = "constant"):
    milestones = [40, 60]
    match scheduler_name.lower():
        case "constant":
            baseline_scheduler = ConstantLR(optimizer, factor=1.0)
        case "multistep":
            baseline_scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        case _:
            baseline_scheduler = ConstantLR(optimizer, factor=1.0)
    return baseline_scheduler
