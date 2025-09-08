import matplotlib.pyplot as plt
import numpy as np
import wandb
from flwr.common import Metrics


def _weighted_average(metrics: list[tuple[int, Metrics]], method_type: str = "fit") -> dict:
    assert method_type in ["fit", "eval"], f"_weighted_average 'method_type' must be 'fit' or 'eval', " \
                                           f"not '{method_type}'"
    cids = [m["client_id"] for _, m in metrics]
    if method_type == "fit":
        # Multiply accuracy of each client by number of examples used
        sum_accuracies_top1 = [num_examples * m["accuracy_top1"] for num_examples, m in metrics]
        accuracies_top1 = [m["accuracy_top1"] for num_examples, m in metrics]
        velocities = [m.get("neve.velocity", -1.0) for _, m in metrics]
        losses = [num_examples * m["loss"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        distributions = [m["data_distribution"] for _, m in metrics if "data_distribution" in m]
        velocities_hist = []
        for _, m in metrics:
            c_keys = [k for k in m.keys() if "velocity_histogram" in k]
            c_keys_layers = list(set([".".join(k.split(".")[2:-1]) for k in c_keys]))
            c_keys_number = []
            for c_key in c_keys_layers:
                current_max = 0
                for key in c_keys:
                    if c_key in key:
                        current_max = max(current_max, int(key.split(".")[-1]))
                c_keys_number.append(current_max + 1)
            velocity_hist = {key: [0 for _ in range(key_len)] for key, key_len in zip(c_keys_layers, c_keys_number)}
            for c_key in c_keys_layers:
                for key in c_keys:
                    if c_key in key:
                        velocity_hist[c_key][int(key.split(".")[-1])] = m[key]
            velocities_hist.append(velocity_hist)
        lrs = [m["lr"] for _, m in metrics]
        # Aggregate and return custom metric (weighted average)
        aggregate_data = {
            "accuracy_top1": sum(sum_accuracies_top1) / sum(examples),
            "loss": sum(losses) / sum(examples),
            "lr": {},
            "clients": {
                cid: {
                    "accuracy_top1": train_acc_t1, "velocity": velocity,
                    "velocity_histogram": {
                        layer_name: wandb.Histogram(
                            np_histogram=np.histogram(np.array(layer_hist), bins=min(32, len(layer_hist))))
                        for layer_name, layer_hist in velocity_hist.items()
                    }
                } for cid, train_acc_t1, velocity, velocity_hist in
                zip(cids, accuracies_top1, velocities, velocities_hist)
            }
        }
        if distributions:
            for cid, distribution in zip(cids, distributions):
                distribution = [float(value) for value in distribution.split(";")]
                classes = [class_id for class_id in range(len(distribution))]
                plt.figure(figsize=(12, 6))
                plt.bar(classes, distribution, color='skyblue')
                plt.xlabel('Classes')
                plt.ylabel('Instances [%]')
                plt.title('Instances/Classes Distribution')
                # Creazione del grafico a barre con matplotlib
                aggregate_data["clients"][cid]["data_distribution"] = wandb.Image(plt,
                                                                                  caption="Instances/Classes Distribution")
                plt.close()
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
        sum_val_accuracies_top1 = [m["size"] * m["accuracy_top1"] for _, m in metrics]
        val_accuracies_top1 = [m["accuracy_top1"] for _, m in metrics]
        val_losses = [m["size"] * m["loss"] for _, m in metrics]
        val_examples = [m["size"] for _, m in metrics]

        # Aggregate and return custom metric (weighted average)
        aggregate_data = {
            "accuracy_top1": sum(sum_val_accuracies_top1) / sum(val_examples),
            "loss": sum(val_losses) / sum(val_examples),
            "clients": {
                cid: {"accuracy_top1": val_acc_t1}
                for cid, val_acc_t1 in zip(cids, val_accuracies_top1)
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
        "clients_train": {},
    }
    # Update each client train accuracy
    for key, data in aggregate_data["clients"].items():
        if key not in data_to_log["clients_train"].keys():
            data_to_log["clients_train"][key] = {}
        data_to_log["clients_train"][key]["train_accuracy_top1"] = data["accuracy_top1"]
        data_to_log["clients_train"][key]["velocity"] = data.get("velocity", -1.0)
        data_to_log["clients_train"][key]["velocity_histogram"] = data.get("velocity_histogram", None)
        if "data_distribution" in data:
            data_to_log["clients_train"][key]["data_distribution"] = data.get("data_distribution", None)
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
                "top1": aggregate_data["accuracy_top1"],
            },
            "loss": aggregate_data["loss"]
        },
        "clients_eval": {},
    }
    # Update each client val & test accuracy
    for key, data in aggregate_data["clients"].items():
        if key not in data_to_log["clients_eval"].keys():
            data_to_log["clients_eval"][key] = {}
        data_to_log["clients_eval"][key]["val_accuracy_top1"] = data["accuracy_top1"]
    #
    wandb.log(data_to_log, commit=False)
    return aggregate_data
