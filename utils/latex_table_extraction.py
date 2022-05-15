from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas import DataFrame
from scipy import stats

import utils

# Table values
latex_caption = "Metrics for Dataset 1 Local Univariate LSTM model"
latex_label = "results:" + latex_caption.replace(" ", "_")
# Metrics average does not currenly work as expected.
metrics_average = False

# Select save location for generated table
dataset = "dataset_seasonal"
table_save_path = f"./MastersThesis/tables/results/{dataset}"
figure_base_path = f"./MastersThesis/figs/results"

# Select projects to be imported
base_path = "./models/"
# Multiple projects does not work?

projects = {
    "sarima": f"{dataset}-sarima",

    "local univariate lstm": f"{dataset}-lstm-local-univariate-tune-400-trials",
    "local multivariate lstm": f"{dataset}-lstm-local-multivariate-tune-400-trials",
    "global univariate lstm": f"{dataset}-lstm-global-univariate-tune-400-trials",
    "global multivariate lstm": f"{dataset}-lstm-global-multivariate-tune-400-trials",

    "local univariate cnn ae lstm": f"{dataset}-cnn-ae-lstm-local-univariate",
    "local multivariate cnn ae lstm": f"{dataset}-cnn-ae-lstm-local-multivariate",
    "global univariate cnn ae lstm": f"{dataset}-cnn-ae-lstm-global-univariate",
    "global multivariate cnn ae lstm": f"{dataset}-cnn-ae-lstm-global-multivariate",
}

# Select metrics to be used
metric_types = ["smape", "mase", "days"]


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
    label = label.replace("_", "-").replace(" ", "-")

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
    label = label.replace("_", "-").replace(" ", "-")

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
    _dataset = dataset.replace("_", "-")
    for exp in projects:
        extract_dataset_metrics_table(
            f"Metrics from experiment, {_dataset}, {exp}",
            f"{exp}-{_dataset}",
            projects[exp],
            base_path,
            table_save_path=table_save_path
        )


def create_shared_avg_table_all_experiments():
    _dataset = dataset.replace("_", "-")
    extract_average_experiment_metrics(
        f"Average values for all experiment for {_dataset}",
        f"Average-metric-{_dataset}",
        projects,
        base_path,
        table_save_path
    )


def list_of_metrics(metrics, metric_name):
    metric_list = []
    for metric in metrics:
        metric_list.append(
            metrics[metric][metric_name]
        )
    return metric_list


def metrics_experiment_lists(experiments: Dict[str, str], metric_name: str= "mase"):
    # Create list with lists of error metrics for each experiment
    updated_metrics_list = []
    updated_metrics_list_names = []
    for experiment_name in experiments:
        path = f"{base_path}{experiments[experiment_name]}/metrics.txt"
        metrics = extract_metrics_from_file(path)
        metric_list = list_of_metrics(metrics, metric_name)
        updated_metrics_list.append(metric_list)
        updated_metrics_list_names.append(experiment_name)
    return updated_metrics_list, updated_metrics_list_names


def freidman_test(experiments: Dict[str, str], metric_name:str="mase"):
    updated_metrics_list, _ = metrics_experiment_lists(experiments, metric_name)
    # Use Freidman test on metrics
    freidman = stats.friedmanchisquare(
        *updated_metrics_list[:-3]
    )
    print(freidman)


def metrics_experiment_box_plot(experiments: Dict[str, str], metric_name: str = "mase"):
    metrics_list, metrics_list_names = metrics_experiment_lists(experiments, metric_name)
    metrics_dict = {}
    for i in range(len(metrics_list)):
        print(len(metrics_list[i]), metrics_list_names[i])
        metrics_dict[metrics_list_names[i]] = metrics_list[i]
    metrics_dataframe = DataFrame(metrics_dict)
    plt.rcParams.update({'font.size': 12})
    ax = sns.boxplot(data=metrics_dataframe)
    plt.xlabel("Experiments")
    plt.ylabel(metric_name)
    ax.set_xticklabels(metrics_list_names, rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(
        f"{figure_base_path}/boxplot/{metric_name}-{dataset}",

    )



# create_all_tables_each_experiment()
# create_shared_avg_table_all_experiments()
# freidman_test(projects, "mase")

metrics_experiment_box_plot(projects, "smape")
