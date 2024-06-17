import numpy as np
import wandb
from flwr.common import Metrics

from src.utils.wandb_figure import create_confusion_matrix_figure, create_distribution_figure
from src.utils.wandb_figure import create_velocity_bar_figure
from src.utils.wandb_figure import mplfig_2_wandbfig


def _weighted_average(metrics: list[tuple[int, Metrics]], method_type: str = "fit") -> Metrics:
    assert method_type in ["fit", "eval"], f"_weighted_average 'method_type' must be 'fit' or 'eval', " \
                                           f"not '{method_type}'"
    if method_type == "fit":
        # Multiply accuracy of each client by number of examples used
        accuracies_top1 = [num_examples * m["accuracy_top1"] for num_examples, m in metrics]
        accuracies_top5 = [num_examples * m["accuracy_top5"] for num_examples, m in metrics]
        losses = [num_examples * m["loss"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        cids = [m["client_id"] for _, m in metrics]
        lrs = [m["lr"] for _, m in metrics]
        # Aggregate and return custom metric (weighted average)
        aggregate_data = {
            "accuracy_top1": sum(accuracies_top1) / sum(examples),
            "accuracy_top5": sum(accuracies_top5) / sum(examples),
            "loss": sum(losses) / sum(examples), "lr": {}
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

        # Plot data distributions
        data_distributions = {}
        for _, m in metrics:
            if "data_distribution_count" not in m or "data_distribution_labels" not in m:
                continue
            data_distributions[m["client_id"]] = (
                np.frombuffer(m["data_distribution_count"], dtype=float).tolist(),
                np.frombuffer(m["data_distribution_labels"], dtype=int).tolist()
            )
        if data_distributions:
            aggregate_data["data_distribution"] = data_distributions
    else:
        # Multiply accuracy of each client by number of examples used
        val_accuracies_top1 = [m["val_size"] * m["val_accuracy_top1"] for _, m in metrics]
        val_accuracies_top5 = [m["val_size"] * m["val_accuracy_top5"] for _, m in metrics]
        val_losses = [m["val_size"] * m["val_loss"] for _, m in metrics]
        val_examples = [m["val_size"] for _, m in metrics]
        test_accuracies_top1 = [m["test_size"] * m["test_accuracy_top1"] for _, m in metrics]
        test_accuracies_top5 = [m["test_size"] * m["test_accuracy_top5"] for _, m in metrics]
        test_losses = [m["test_size"] * m["test_loss"] for _, m in metrics]
        test_examples = [m["test_size"] for _, m in metrics]

        # Plot confusion matrix
        avg_matrix = np.sum([np.frombuffer(m["confusion_matrix"], dtype=float).reshape((m["confusion_matrix_shape_d0"],
                                                                                        m["confusion_matrix_shape_d1"]))
                             for _, m in metrics], axis=0) / len(metrics)

        avg_confusion_matrix_fig = create_confusion_matrix_figure(avg_matrix, title="Average Test Confusion Matrix")
        avg_confusion_matrix_plt = mplfig_2_wandbfig(avg_confusion_matrix_fig)

        # Aggregate and return custom metric (weighted average)
        aggregate_data = {
            "val_accuracy_top1": sum(val_accuracies_top1) / sum(val_examples),
            "val_accuracy_top5": sum(val_accuracies_top5) / sum(val_examples),
            "val_loss": sum(val_losses) / sum(val_examples),
            "test_accuracy_top1": sum(test_accuracies_top1) / sum(test_examples),
            "test_accuracy_top5": sum(test_accuracies_top5) / sum(test_examples),
            "test_loss": sum(test_losses) / sum(test_examples),
            'avg_confusion_matrix': avg_confusion_matrix_plt
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
                "top5": aggregate_data["accuracy_top5"]
            },
            "loss": aggregate_data["loss"]},
        "lr": {client_id: lr for client_id, lr in aggregate_data["lr"].items()}
    }
    for _, client_data in metrics:
        if "neve.continue_training" in client_data.keys():
            if "neve" not in data_to_log.keys():
                data_to_log["neve"] = {}
            data_to_log["neve"][client_data["client_id"]] = {
                "continue_training": 1 if client_data["neve.continue_training"] else 0,
                "model_value": client_data["neve.model_value"],
                "model_avg_value": client_data["neve.model_avg_value"],
                "neurons_velocity": {}
            }
            if "neve.model_mse_value" in client_data.keys():
                data_to_log["neve"][client_data["client_id"]]["model_mse_value"] = client_data["neve.model_mse_value"]

            for key, value in client_data.items():
                if not key.startswith("neve.neurons_velocity."):
                    continue
                layer_name = key.split(".")[-1]
                np_data = np.frombuffer(value, dtype=np.float32)
                np_neurons = [val for val in range(0, np_data.shape[0])]

                distribution_fig = create_velocity_bar_figure(np_data, np_neurons,
                                                              title=f"Client: {client_data['client_id']} - Per Neuron "
                                                                    f"Velocity - Layer: {layer_name}")
                neurons_fig = mplfig_2_wandbfig(distribution_fig)
                data_to_log["neve"][client_data["client_id"]]["neurons_velocity"][f"{layer_name}"] = neurons_fig
    #
    if "data_distribution" in aggregate_data:
        for client_id, distribution_data in aggregate_data["data_distribution"].items():
            if "data_distribution" not in data_to_log:
                data_to_log["data_distribution"] = {}
            client_distr, client_labels = distribution_data
            # Create client classes distribution plot
            distribution_fig = create_distribution_figure(client_distr, client_labels,
                                                          title=f"Client.{str(client_id)} Train-set Distribution")
            distribution_plt = mplfig_2_wandbfig(distribution_fig)
            data_to_log["data_distribution"][f"{str(client_id)}"] = distribution_plt
    wandb.log(data_to_log, commit=False)
    return aggregate_data


def weighted_average_eval(metrics: list[tuple[int, Metrics]]) -> Metrics:
    print("Eval data to aggregate:", metrics)
    aggregate_data = _weighted_average(metrics, method_type="eval")
    print("Eval aggregation result:", aggregate_data)
    wandb.log(
        {
            "val": {
                "accuracy": {
                    "top1": aggregate_data["val_accuracy_top1"],
                    "top5": aggregate_data["val_accuracy_top5"]
                },
                "loss": aggregate_data["val_loss"]
            },
            "test": {
                "accuracy": {
                    "top1": aggregate_data["test_accuracy_top1"],
                    "top5": aggregate_data["test_accuracy_top5"]
                },
                "loss": aggregate_data["test_loss"]
            },
            "confusion_matrix": aggregate_data["avg_confusion_matrix"]
        }
    )
    return aggregate_data
