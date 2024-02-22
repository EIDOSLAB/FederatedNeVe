import os

import pandas as pd


def read_lr_csv(lr_csv_path, threshold: float = 1e-3, max_epochs: int = 250 - 1, metrics=None):
    if metrics is None:
        metrics = []
    # read lr csv
    df_lr = pd.read_csv(lr_csv_path)
    # find run names
    runs = []
    step_column = "Step"
    for idx, column in enumerate(df_lr.columns):
        if idx == 0:
            step_column = column
            continue  # Step
        if (idx + 2) % 3 == 0:
            runs.append(column)
    print(runs)
    # find Step where lr is lower than threshold
    runs_data = {}
    for run in runs:
        if len(df_lr[df_lr[run] < threshold][step_column].nsmallest(1).values) > 0:
            run_epoch = df_lr[df_lr[run] < threshold][step_column].nsmallest(1).values[0]
        else:
            run_epoch = max_epochs
        run_data = {metric: 0.0 for metric in metrics}
        run_data["epoch"] = run_epoch
        runs_data[run] = run_data

    # Calculate epoch avg + dev.std
    print("Stopping epochs:", " ".join(str(runs_data[run]["epoch"]) for run in runs))
    avg_epoch = sum([runs_data[run]["epoch"] for run in runs]) / len(runs_data)
    std_epoch = (sum([(runs_data[run]["epoch"] - avg_epoch) ** 2 for run in runs]) / (len(runs_data) - 1)) ** 0.5
    return df_lr.columns[0], avg_epoch, std_epoch, runs, runs_data


def elaborate_metric_data(metric_file_path, value_multiplier, print_name, runs, runs_data, step_column):
    # read accuracy dataframe
    df_acc = pd.read_csv(metric_file_path)
    # rename runs names
    for idx, run in enumerate(runs):
        run_name = run.split(" - ")[0]
        for column in df_acc.columns:
            if run_name in column:
                runs[idx] = column
                runs_data[column] = runs_data[run]
                break

    for run in runs:
        run_data = runs_data[run]
        run_data[print_name] = round(
            df_acc.loc[df_acc[step_column] == run_data["epoch"]][run].values[0] * value_multiplier, 4)

    avg_acc = round(sum([runs_data[run][print_name] for run in runs]) / len(runs), 4)
    std_acc = (sum([(runs_data[run][print_name] - avg_acc) ** 2 for run in runs]) / (len(runs) - 1)) ** 0.5
    print(f"{print_name}:", f"{avg_acc:3.4f}", "±", f"{std_acc:2.4f}", )


def main():
    # settings
    threshold = 1e-3
    max_epochs = 250 - 1

    base = "C:\\Users\\Gianluca\\Downloads"
    lr_csv = os.path.join(base, "lr.csv")
    metrics_csv_names = [("train_acc.csv", 100, "Train Accuracy"), ("train_loss.csv", 1, "Train Loss"),
                         ("val_acc.csv", 100, "Validation Accuracy"), ("val_loss.csv", 1, "Validation Loss"),
                         ("test_acc.csv", 100, "Test Accuracy"), ("test_loss.csv", 1, "Test Loss")]
    step_column, avg_epoch, std_epoch, runs, runs_data = read_lr_csv(lr_csv, threshold=threshold, max_epochs=max_epochs,
                                                                     metrics=[metric_name for _, _, metric_name in
                                                                              metrics_csv_names])

    print("Avg. stopping epoch:", int(avg_epoch), "±", int(std_epoch))
    for metric_file_name, value_multiplier, print_name in metrics_csv_names:
        if os.path.exists(os.path.join(base, metric_file_name)):
            elaborate_metric_data(os.path.join(base, metric_file_name), value_multiplier, print_name,
                                  runs, runs_data, step_column)
        else:
            print(f"Skipping: '{metric_file_name}'. File not found.")


if __name__ == "__main__":
    main()
