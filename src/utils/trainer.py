import datetime
import time

import torch

from NeVe import NeVeOptimizer
from utils.metrics import Accuracy, AverageMeter


def train_epoch(model: torch.nn.Module, data_loaders: dict, optimizer, scheduler, grad_scaler, device: str,
                amp: bool = True, epoch: int = 0) -> dict:
    epoch_logs = {}

    # Training phase
    if "train" in data_loaders.keys() and data_loaders["train"]:
        epoch_logs["train"] = run(model, data_loaders["train"], optimizer, grad_scaler, device, amp, epoch, "Train")
        if not isinstance(scheduler, NeVeOptimizer):
            scheduler.step()

    # Validation phase
    if "val" in data_loaders.keys() and data_loaders["val"]:
        epoch_logs["val"] = run(model, data_loaders["val"], None, grad_scaler, device, amp, epoch, "Validation")

    # Test phase
    if "test" in data_loaders.keys() and data_loaders["test"]:
        epoch_logs["test"] = run(model, data_loaders["test"], None, grad_scaler, device, amp, epoch, "Test")

    # NeVe phase
    if "aux" in data_loaders.keys() and data_loaders["aux"] and isinstance(scheduler, NeVeOptimizer):
        with scheduler:
            _ = run(model, data_loaders["aux"], None, grad_scaler, device, amp, epoch, "Aux")
        epoch_logs["neve"] = scheduler.step()

    epoch_logs["lr"] = optimizer.param_groups[0]["lr"]
    return epoch_logs


def run(model, dataloader, optimizer, scaler, device, amp, epoch, run_type):
    acc = Accuracy((1, 5))

    accuracy_meter_1 = AverageMeter()
    accuracy_meter_5 = AverageMeter()
    loss_meter = AverageMeter()
    batch_time = AverageMeter()

    train = optimizer is not None
    model.train(train)

    if train:
        optimizer.zero_grad()

    t1 = time.time()

    # Define the loss function
    loss_fn = torch.nn.functional.cross_entropy

    for batch, (images, target) in enumerate(dataloader):
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=(device == "cuda" and amp)):
                output = model(images)
                loss = loss_fn(output, target)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        accuracy = acc(output, target)
        accuracy_meter_1.update(accuracy[0].item(), target.shape[0])
        accuracy_meter_5.update(accuracy[1].item(), target.shape[0])
        loss_meter.update(loss.item(), target.shape[0])

        batch_time.update(time.time() - t1)
        t1 = time.time()
        eta = batch_time.avg * (len(dataloader) - batch)

        print(f"{run_type}: [{epoch}][{batch + 1}/{len(dataloader)}]:\t"
              f"BT {batch_time.avg:.3f}\t"
              f"ETA {datetime.timedelta(seconds=eta)}\t"
              f"loss {loss_meter.avg:.3f}\t")
    return {
        'loss': loss_meter.avg,
        'accuracy': {
            "top1": accuracy_meter_1.avg / 100,
            "top5": accuracy_meter_5.avg / 100
        }
    }
