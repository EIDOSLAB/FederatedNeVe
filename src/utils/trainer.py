import datetime
import time
from collections import Counter

import torch

from NEq.scheduler import NEqScheduler
from src.NeVe.scheduler import NeVeScheduler
from src.NeVe.utils.data import NeVeData
from src.utils.metrics import Accuracy, AverageMeter


def train_epoch(model: torch.nn.Module, data_loaders: dict, optimizer, scheduler, grad_scaler, device: str,
                amp: bool = True, epoch: int = 0) -> tuple[dict, NeVeData | None, dict]:
    epoch_logs = {"lr": optimizer.param_groups[0]["lr"]}
    # Training phase
    if "train" in data_loaders.keys() and data_loaders["train"]:
        epoch_logs["train"] = run(model, data_loaders["train"], optimizer, grad_scaler, device, amp, epoch, "Train")
        if not (isinstance(scheduler, NeVeScheduler) or isinstance(scheduler, NEqScheduler)):
            scheduler.step()

    # Validation phase
    if "val" in data_loaders.keys() and data_loaders["val"]:
        epoch_logs["val"] = run(model, data_loaders["val"], None, grad_scaler, device, amp, epoch, "Validation")

    # Test phase
    if "test" in data_loaders.keys() and data_loaders["test"]:
        epoch_logs["test"] = run(model, data_loaders["test"], None, grad_scaler, device, amp, epoch, "Test")

    # NeVe phase
    neve_data = None
    neq_data = None
    if "aux" in data_loaders.keys() and data_loaders["aux"]:
        with scheduler:
            _ = run(model, data_loaders["aux"], None, grad_scaler, device, amp, epoch, "Aux")
        if isinstance(scheduler, NeVeScheduler):
            neve_data = scheduler.step()
            if neve_data:
                epoch_logs["aux"] = neve_data.as_dict
        elif isinstance(scheduler, NEqScheduler):
            neq_data = scheduler.step()
            if neq_data:
                epoch_logs["neq"] = neq_data

    return epoch_logs, neve_data, neq_data


def run(model: torch.nn.Module, dataloader, optimizer, scaler, device: str, amp: bool = True,
        epoch: int = 0, run_type: str = "Train"):
    acc = Accuracy((1,))  # 5))

    accuracy_meter_1 = AverageMeter()
    # accuracy_meter_5 = AverageMeter()
    loss_meter = AverageMeter()
    batch_time = AverageMeter()

    train = optimizer is not None
    model.train(train)

    if train:
        optimizer.zero_grad()

    t1 = time.time()

    # Define the loss function
    loss_fn = torch.nn.functional.cross_entropy

    confusion_matrix = None
    labels_counter = None

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

        # Init confusion matrix and labels counter
        if confusion_matrix is None:
            confusion_matrix = torch.zeros(output.size(1), output.size(1), dtype=torch.int64, device=device)

        if labels_counter is None:
            labels_counter = Counter({label: 0 for label in range(output.size(1))})

        # Update labels counter
        labels_counter.update(target.cpu().numpy())
        # Update confusion matrix
        _, preds = torch.max(output, 1)
        # Aggiorna la matrice di confusione
        for t, p in zip(target.view(-1), preds.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

        accuracy = acc(output, target)
        accuracy_meter_1.update(accuracy[0].item(), target.shape[0])
        # accuracy_meter_5.update(accuracy[1].item(), target.shape[0])
        loss_meter.update(loss.item(), target.shape[0])

        batch_time.update(time.time() - t1)
        t1 = time.time()
        eta = batch_time.avg * (len(dataloader) - batch)

        print(f"{run_type}: [{epoch}][{batch + 1}/{len(dataloader)}]:\t"
              f"BT {batch_time.avg:.3f}\t"
              f"ETA {datetime.timedelta(seconds=eta)}\t"
              f"Accuracy {accuracy_meter_1.avg:.3f}\t"
              f"Batch Loss {loss.item():.3f}\t"
              f"Cumulative Loss {loss_meter.avg:.3f}\t")

    # Normalize confusion matrix
    confusion_matrix = confusion_matrix.cpu().numpy().astype(float)
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    confusion_matrix = confusion_matrix / row_sums

    return {
        "loss": loss_meter.avg,
        "accuracy": {
            "top1": accuracy_meter_1.avg / 100,
            # "top5": accuracy_meter_5.avg / 100
        },
        "confusion_matrix": confusion_matrix,
        "labels_counter": labels_counter
    }
