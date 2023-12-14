from flwr.common import Metrics


def _weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    aggregate_data = {"accuracy": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples)}
    return aggregate_data


def weighted_average_fit(metrics: list[tuple[int, Metrics]]) -> Metrics:
    print("Fit data to aggregate:", metrics)
    aggregate_data = _weighted_average(metrics)
    print("Fit aggregation result:", aggregate_data)
    return aggregate_data


def weighted_average_eval(metrics: list[tuple[int, Metrics]]) -> Metrics:
    print("Eval data to aggregate:", metrics)
    aggregate_data = _weighted_average(metrics)
    print("Eval aggregation result:", aggregate_data)
    return aggregate_data
