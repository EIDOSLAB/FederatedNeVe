import datetime
import time

import torch

from my_federated.NeVe.scheduler import NeVeScheduler
from my_federated.NeVe.utils.data import NeVeData
from my_federated.utils.metrics import Accuracy, AverageMeter


def train_epoch(model: torch.nn.Module, data_loaders: dict, task_name: str, optimizer, scheduler, neve_scheduler,
                grad_scaler, device: str, amp: bool = True, epoch: int = 0) -> tuple[dict, NeVeData | None]:
    epoch_logs = {"lr": optimizer.param_groups[0]["lr"]}
    # Training phase
    if "train" in data_loaders.keys() and data_loaders["train"]:
        epoch_logs["train"] = train_model(model, data_loaders["train"], task_name, optimizer, grad_scaler, device,
                                          amp, epoch)
        scheduler.step()

    # Validation phase
    if "val" in data_loaders.keys() and data_loaders["val"]:
        epoch_logs["val"] = eval_model(model, data_loaders["val"], task_name, device, amp, epoch,
                                       "Validation")

    # Test phase
    if "test" in data_loaders.keys() and data_loaders["test"]:
        epoch_logs["test"] = eval_model(model, data_loaders["test"], task_name, device, amp, epoch, "Test")

    # NeVe phase
    neve_data = None
    if "aux" in data_loaders.keys() and data_loaders["aux"]:
        if isinstance(neve_scheduler, NeVeScheduler):
            with neve_scheduler:
                _ = eval_model(model, data_loaders["aux"], task_name, device, amp, epoch, "Aux")
                neve_data = neve_scheduler.step()
                if neve_data:
                    epoch_logs["aux"] = neve_data.as_dict

    return epoch_logs, neve_data


def train_model(model: torch.nn.Module, dataloader, task_name: str, optimizer, scaler, device: str, amp: bool = True,
                epoch: int = 0, print_batch: int = 100):
    accuracy_evaluator = Accuracy((1,), task=task_name)

    batch_time = AverageMeter()
    loss_avg, accuracy_1_avg, balanced_accuracy_1_avg, auc_avg = 0.0, 0.0, 0.0, 0.0

    model.train()
    optimizer.zero_grad()

    t1 = time.time()

    # Define the loss function
    loss_fn = get_loss_fn(task_name, device)

    for batch, (images, target) in enumerate(dataloader):
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
        target = preprocess_target(target, task_name)

        with torch.amp.autocast("cuda", enabled=(device == "cuda" and amp)):
            output = model(images)
            loss = loss_fn(output, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        _, preds = torch.max(output, 1)

        target = postprocess_target(target, task_name)
        accuracy_1_avg, balanced_accuracy_1_avg, auc_avg, loss_avg = accuracy_evaluator(output.detach().cpu().numpy(),
                                                                                        target.detach().cpu().numpy(),
                                                                                        loss)

        batch_time.update(time.time() - t1)
        t1 = time.time()
        eta = batch_time.avg * (len(dataloader) - batch)
        if batch % print_batch == 0:
            print(f"Train: [{epoch}][{batch + 1}/{len(dataloader)}]:\t"
                  f"BT {batch_time.avg:.3f}\t"
                  f"ETA {datetime.timedelta(seconds=eta)}\t"
                  f"Batch Loss {loss.item():.3f}\t")
        #
    print(f"Train: [{epoch}]:\t"
          f"Accuracy {accuracy_1_avg:.3f}\t"
          f"Balanced Accuracy {balanced_accuracy_1_avg:.3f}\t"
          f"AUC {auc_avg:.3f}\t"
          f"Cumulative Loss {loss_avg:.3f}\t")
    #
    return {
        "loss": loss_avg,
        "accuracy": {
            "top1": accuracy_1_avg,
        },
        "balanced_accuracy": {
            "top1": balanced_accuracy_1_avg,
        },
        "auc": auc_avg,
    }


def eval_model(model: torch.nn.Module, dataloader, task_name: str, device: str, amp: bool = True, epoch: int = 0,
               run_type: str = "Train", print_batch: int = 50):
    accuracy_evaluator = Accuracy((1,), task=task_name)  # 5))

    batch_time = AverageMeter()
    loss_avg, accuracy_1_avg, balanced_accuracy_1_avg, auc_avg = 0.0, 0.0, 0.0, 0.0

    model.eval()

    t1 = time.time()

    # Define the loss function
    loss_fn = get_loss_fn(task_name, device)

    for batch, (images, target) in enumerate(dataloader):
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
        target = preprocess_target(target, task_name)

        with torch.amp.autocast("cuda", enabled=(device == "cuda" and amp)):
            output = model(images)
            loss = loss_fn(output, target)

        _, preds = torch.max(output, 1)

        target = postprocess_target(target, task_name)
        accuracy_1_avg, balanced_accuracy_1_avg, auc_avg, loss_avg = accuracy_evaluator(output.detach().cpu().numpy(),
                                                                                        target.detach().cpu().numpy(),
                                                                                        loss)

        batch_time.update(time.time() - t1)
        t1 = time.time()
        eta = batch_time.avg * (len(dataloader) - batch)
        if batch % print_batch == 0:
            print(f"{run_type}: [{epoch}][{batch + 1}/{len(dataloader)}]:\t"
                  f"BT {batch_time.avg:.3f}\t"
                  f"ETA {datetime.timedelta(seconds=eta)}\t"
                  f"Batch Loss {loss.item():.3f}\t")
        #
    print(f"{run_type}: [{epoch}]:\t"
          f"Accuracy {accuracy_1_avg:.3f}\t"
          f"Balanced Accuracy {balanced_accuracy_1_avg:.3f}\t"
          f"AUC {auc_avg:.3f}\t"
          f"Cumulative Loss {loss_avg:.3f}\t")
    #
    return {
        "loss": loss_avg,
        "accuracy": {
            "top1": accuracy_1_avg,
        },
        "balanced_accuracy": {
            "top1": balanced_accuracy_1_avg,
        },
        "auc": auc_avg,
    }


def get_loss_fn(task_name, device):
    if task_name in ["binary-class", "multi-class", "ordinal-regression"]:
        return torch.nn.CrossEntropyLoss()
    else:
        return torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([0.95]).to(device), reduction="mean")


def preprocess_target(target, task_name):
    if task_name in ["binary-class", "multi-class", "ordinal-regression"]:
        return target.squeeze().long()
    else:
        return target.float()


def postprocess_target(target, task_name):
    if task_name in ["binary-class", "multi-class", "ordinal-regression"]:
        return target.float().resize_(len(target), 1)
    else:
        return target