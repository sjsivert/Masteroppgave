import numpy as np
from pandas import DataFrame

import utils

# Table values
latex_caption = "Metrics for Dataset 1 Local Univariate LSTM model"
latex_label = "results:" + latex_caption.replace(" ", "_")
# Metrics average does not currenly work as expected.
metrics_average = False

# Select save location for generated table
table_save_path = "./MastersThesis/tables/results/"

# Select projects to be imported
base_path = "./models/"
# Multiple projects does not work?
projects = {
    "arima": "dataset_1_arima",
    "local_univariate_lstm": "dataset_1-lstm-local-unviariate-tune-400-trials",
    "local multivariate lstm": "dataset_1-lstm-multivariate-tune-400-trails",
    "global univariate lstm": "dataset_1-lstm-global-unviariate-tune-400-trials",
    "global multivariate lstm": "dataset_1-lstm-global-multivariate-tune-400-trials",
}

# Select metrics to be used
metric_types = ["mase", "smape", "days"]


# Read metrics from txt file
def extract_metrics_from_file(path) -> DataFrame:
    scores = {}
    file_content = None
    with open(path, "r") as f:
        f.readline()  # Remove first empty line
        file_content = f.read()
    file_content = file_content.split("\n\n")
    for data_set_metrics in file_content:
        split_metrics = data_set_metrics.split("\n")
        data_set_name = split_metrics[0]
        if "Average" in data_set_name:
            continue
        data_set_name = data_set_name.split("-")[1].replace("_", "")
        metrics = {}
        for metric in split_metrics[1:]:
            metric_split = metric.split(": ")
            metric_name_split, metric_value = metric_split[0], metric_split[1]
            metric_name = metric_name_split.rsplit("_", 1)[-1].lower()
            if metric_name not in metric_types or metric_name in metrics:
                continue
            if "days" in metric_name.lower():
                metric_name = metric_name_split.split("_", 1)[-1].replace("_", "-").lower()
                metric_name = metric_name.rsplit("-", 1)[0]

            metric_value = float(metric_value)
            metrics[metric_name] = round(metric_value, 3)
        scores[data_set_name] = metrics
    return scores


# Create table of time series and metrics. This is for one model and dataset
def extract_dataset_metrics_table(caption, label, experiment, base_path, table_save_path):
    caption = caption.replace("_", "-")
    label = label.replace("_", "-")

    metrics = extract_metrics_from_file(f"{base_path}{experiment}/metrics.txt")
    metrics = DataFrame(metrics).transpose()
    metrics["dataset"] = metrics.index
    metrics.set_index("dataset", inplace=True)

    utils.dataframe_to_latex_tabular(
        metrics, caption, label, add_index=True, save_local=table_save_path
    )


# Create table for average, given experimens and metrics
def extract_average_experiment_metrics(caption, label, experiments, base_path, table_save_path):
    caption = caption.replace("_", "-")
    label = label.replace("_", "-")

    updated_metrics = {}
    for experiment_name in experiments:
        path = f"{base_path}{experiments[experiment_name]}/metrics.txt"
        metrics = extract_metrics_from_file(path)
        average_metrics = calc_average_metrics(metrics)
        updated_metrics[experiment_name] = average_metrics
    updated_metrics = DataFrame(updated_metrics).transpose()
    updated_metrics["Experiment"] = updated_metrics.index
    updated_metrics.set_index("Experiment", inplace=True)
    utils.dataframe_to_latex_tabular(
        updated_metrics, caption, label, add_index=True, save_local=table_save_path
    )


def calc_average_metrics(metrics, calc_type="average"):
    calc = {}
    for dataset, metrics_dict in metrics.items():
        for metric_name, metric_value in metrics_dict.items():
            if metric_name not in calc:
                calc[metric_name] = []
            calc[metric_name].append(metric_value)
    # Calculate average, mean or std
    for metric_name, metrics_list in calc.items():
        if calc_type == "average":
            val = np.average(metrics_list)
        elif calc_type == "mean":
            val = np.mean(metrics_list)
        elif calc_type == "std":
            val = np.std(metrics_list)
        else:
            val = {"Placeholder": 0}
        calc[metric_name] = round(val, 3)
    return calc


#extract_metrics_from_file(f"{base_path}{projects['local univariate lstm']}/metrics.txt")


def create_all_tables_each_experiment():
    dataset = "dataset-1"
    for exp in projects:
        extract_dataset_metrics_table(
            f"Metrics from experiment, {dataset}, {exp}",
            f"{exp}-{dataset}",
            projects[exp],
            base_path,
            table_save_path=table_save_path
        )


def create_shared_avg_table_all_experiments():
    dataset = "dataset 1"
    extract_average_experiment_metrics(
        f"Average values for all experiment for {dataset}",
        f"Average-metric-{dataset}",
        projects,
        base_path,
        table_save_path
    )



# create_all_tables_each_experiment()
create_shared_avg_table_all_experiments()
