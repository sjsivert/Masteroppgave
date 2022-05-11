import math
from typing import Dict
import os
import csv


def load_dict(path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found at path: {path}")
    with open(path, "r") as f:
        tuning_metrics = {}
        reader = csv.reader(f, delimiter=";")
        next(reader)  # Skip row with column names
        for row in reader:
            if row[0] not in tuning_metrics:
                tuning_metrics[row[0]] = {}
            # Convert from string to dict of string float set values
            error_metrics_set_list = dict(
                (i, float(j)) for i, j in [x.split(":") for x in row[2].split(",")]
            )
            tuning_metrics[row[0]][row[1]] = error_metrics_set_list
        return tuning_metrics


def sort_dict(metrics: Dict, metric_name: str) -> Dict:
    sorted_metrics = {}
    for dataset_name, param_metrics in metrics.items():
        sorted_param_dict = dict(
            sorted(
                param_metrics.items(),
                key=lambda x: float("inf") if math.isnan(x[1][metric_name]) else x[1][metric_name],
                reverse=False,
            )
        )
        sorted_metrics[dataset_name] = sorted_param_dict
    return sorted_metrics


def store_sorted_metrics(metrics: Dict, path):
    with open(path, "w") as f:
        for dataset_name, param_metrics in metrics.items():
            f.write(f"Model: {dataset_name}\n")
            for param, metric_set in param_metrics.items():
                metric_str = ", ".join([f"{x}:{str(y)}" for x, y in metric_set.items()])
                f.write(f"{param} -> {metric_str} \n")
            f.write("\n")


"""
___ Sorting tuning parameters by error function ___
Define the experiment name (local -> Neptune support is added later if needed)
Define the name of the metric to be sorted by, and a new .txt file is created with a sorted value.
"""
if __name__ == "__main__":
    metric_name = "MASE"
    experiment_name = "arima_corr_20_tuning_part2"
    metrics_file_path = f"./models/{experiment_name}/logging/tuning_metrics.csv"
    sorted_save_path = f"./models/{experiment_name}/tuning_{metric_name}.txt"

    metrics_dict = load_dict(metrics_file_path)
    sorted_metrics_dict = sort_dict(metrics_dict, metric_name)
    store_sorted_metrics(sorted_metrics_dict, sorted_save_path)
