import wandb
from flwr.common import Metrics


def _weighted_average(metrics: list[tuple[int, Metrics]], method_type: str = "fit") -> dict:
    assert method_type in ["fit", "eval"], f"_weighted_average 'method_type' must be 'fit' or 'eval', " \
                                           f"not '{method_type}'"
    cids = [m["client_id"] for _, m in metrics]
    if method_type == "fit":
        # Multiply accuracy of each client by number of examples used
        accuracies_top1 = [num_examples * m["accuracy_top1"] for num_examples, m in metrics]
        losses = [num_examples * m["loss"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        lrs = [m["lr"] for _, m in metrics]
        # Aggregate and return custom metric (weighted average)
        aggregate_data = {
            "accuracy_top1": sum(accuracies_top1) / sum(examples),
            "loss": sum(losses) / sum(examples),
            "lr": {},
            "clients": {
                cid: {"train_accuracy_top1": train_acc_t1} for cid, train_acc_t1 in zip(cids, accuracies_top1)
            }
        }
        for client_id, lr in zip(cids, lrs):
            print("LR:", client_id, lr)
            aggregate_data["lr"][str(client_id)] = lr
        # NeVe check:
        is_neve_used = False
        for num_examples, m in metrics:
            if "neve_optimizer" in m.keys():
                is_neve_used = True
                break
        if is_neve_used:
            aggregate_data["neve_optimizer"] = {}
            neve_datas = [m["neve_optimizer"] for _, m in metrics]
            for client_id, neve_data in zip(cids, neve_datas):
                print("Neve Data:", client_id, neve_data)
                aggregate_data["neve_optimizer"][str(client_id)] = neve_data
    else:
        # Multiply accuracy of each client by number of examples used
        val_accuracies_top1 = [m["val_size"] * m["val_accuracy_top1"] for _, m in metrics]
        val_losses = [m["val_size"] * m["val_loss"] for _, m in metrics]
        val_examples = [m["val_size"] for _, m in metrics]
        test_accuracies_top1 = [m["test_size"] * m["test_accuracy_top1"] for _, m in metrics]
        test_losses = [m["test_size"] * m["test_loss"] for _, m in metrics]
        test_examples = [m["test_size"] for _, m in metrics]

        # Aggregate and return custom metric (weighted average)
        aggregate_data = {
            "val_accuracy_top1": sum(val_accuracies_top1) / sum(val_examples),
            "val_loss": sum(val_losses) / sum(val_examples),
            "test_accuracy_top1": sum(test_accuracies_top1) / sum(test_examples),
            "test_loss": sum(test_losses) / sum(test_examples),
            "clients": {
                cid: {"val_accuracy_top1": val_acc_t1, "test_accuracy_top1": test_acc_t1}
                for cid, val_acc_t1, test_acc_t1 in zip(cids, val_accuracies_top1, test_accuracies_top1)
            }
        }
    return aggregate_data


def weighted_average_fit(metrics: list[tuple[int, Metrics]]) -> Metrics:
    print("Fit data to aggregate:", metrics)
    aggregate_data = _weighted_average(metrics, method_type="fit")
    print("Fit aggregation result:", aggregate_data)
    data_to_log = {
        "train": {
            "accuracy": {
                "top1": aggregate_data["accuracy_top1"],
            },
            "loss": aggregate_data["loss"]},
        "lr": {client_id: lr for client_id, lr in aggregate_data["lr"].items()},
        "clients": {},
    }
    # Update each client train accuracy
    for key, data in aggregate_data["clients"].items():
        if key not in data_to_log["clients"].keys():
            data_to_log["clients"][key] = {}
        data_to_log["clients"][key]["train_accuracy_top1"] = data["train_accuracy_top1"]
    # Update each client neve data
    for _, client_data in metrics:
        if "neve.continue_training" in client_data.keys():
            if "neve" not in data_to_log.keys():
                data_to_log["neve"] = {}
            data_to_log["neve"][client_data["client_id"]] = {
                "continue_training": 1 if client_data["neve.continue_training"] else 0,
                "model_value": client_data["neve.model_value"],
                "model_avg_value": client_data["neve.model_avg_value"],
            }
            if "neve.model_mse_value" in client_data.keys():
                data_to_log["neve"][client_data["client_id"]]["model_mse_value"] = client_data["neve.model_mse_value"]
    #
    wandb.log(data_to_log, commit=False)
    return aggregate_data


def weighted_average_eval(metrics: list[tuple[int, Metrics]]) -> Metrics:
    print("Eval data to aggregate:", metrics)
    aggregate_data = _weighted_average(metrics, method_type="eval")
    print("Eval aggregation result:", aggregate_data)
    data_to_log = {
        "val": {
            "accuracy": {
                "top1": aggregate_data["val_accuracy_top1"],
            },
            "loss": aggregate_data["val_loss"]
        },
        "test": {
            "accuracy": {
                "top1": aggregate_data["test_accuracy_top1"],
            },
            "loss": aggregate_data["test_loss"]
        },
        "clients": {},
    }
    # Update each client val & test accuracy
    for key, data in aggregate_data["clients"].items():
        if key not in data_to_log["clients"].keys():
            data_to_log["clients"][key] = {}
        data_to_log["clients"][key]["val_accuracy_top1"] = data["val_accuracy_top1"]
        data_to_log["clients"][key]["test_accuracy_top1"] = data["test_accuracy_top1"]
    wandb.log(
        data_to_log
    )
    return aggregate_data
