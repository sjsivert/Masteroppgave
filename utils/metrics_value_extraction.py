from ipaddress import collapse_addresses
from typing import Dict, List

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
dataset = "dataset_2"
table_save_path = f"./MastersThesis/tables/results/{dataset}"
figure_base_path = f"./MastersThesis/figs/results"

# Select projects to be imported
base_path = "./models/"
# Multiple projects does not work?


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
            if metric_name not in metric_types or metric_name.lower() in [x.lower() for x in metrics.keys()]:
                continue
            if "days" in metric_name.lower():
                metric_name = metric_name_split.split("_", 1)[-1].replace("_", "-").lower()
                metric_name = metric_name.rsplit("-", 1)[0]

            metric_value = float(metric_value)
            metric_name = metric_renaming(metric_name) 
            metrics[metric_name] = round(metric_value, 3)
        scores[data_set_name] = metrics
    return scores

def metric_renaming(metric_name):
    metric_correct_naming = ["sMAPE", "MASE", "MASE-7"]
    i = [x.lower() for x in metric_correct_naming].index(metric_name)
    return metric_correct_naming[i]

# Create table of time series and metrics. This is for one model and dataset
def extract_dataset_metrics_table(caption, label, experiment, base_path, table_save_path):
    caption = caption.replace("_", "-")
    label = label.replace("_", "-").replace(" ", "-")

    metrics = extract_metrics_from_file(f"{base_path}{experiment}/metrics.txt")
    metrics = DataFrame(metrics).transpose()
    metrics["Category ID"] = metrics.index
    metrics.set_index("Category ID", inplace=True)

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




def create_all_tables_each_experiment(projects, save_path=table_save_path, dataset=dataset):
    _dataset = dataset.replace("_", "-")
    for exp in projects:
        extract_dataset_metrics_table(
            f"Metrics from experiment, {_dataset}, {exp}",
            f"{exp}-{_dataset}",
            projects[exp],
            base_path,
            table_save_path=save_path
        )


def create_shared_avg_table_all_experiments(projects, save_path=table_save_path, dataset=dataset):
    _dataset = dataset.replace("_", "-")
    extract_average_experiment_metrics(
        f"Average values for all experiment for {_dataset}",
        f"Average-metric-{_dataset}",
        projects,
        base_path,
        save_path
    )


def list_of_metrics(metrics, metric_name):
    metric_list = []
    for metric in metrics:
        metric_list.append(
            metrics[metric][metric_name]
        )
    return metric_list


def metrics_experiment_lists(experiments: Dict[str, str], metric_name: str= "MASE"):
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


def freidman_test(experiments: Dict[str, str], metric_name:str="MASE"):
    updated_metrics_list, _ = metrics_experiment_lists(experiments, metric_name)
    # Use Freidman test on metrics
    freidman = stats.friedmanchisquare(
        *updated_metrics_list[:-3]
    )
    print(freidman)


def metrics_experiment_box_plot(experiments: Dict[str, str], metric_name: str = "MASE", save_path=figure_base_path, dataset=dataset):
    metrics_list, metrics_list_names = metrics_experiment_lists(experiments, metric_name)
    metrics_dict = {}
    for i in range(len(metrics_list)):
        metrics_dict[metrics_list_names[i]] = metrics_list[i]
    metrics_dataframe = DataFrame(metrics_dict)
    plt.clf()
    plt.cla()
    plt.rcParams.update({'font.size': 12})
    ax = sns.boxplot(data=metrics_dataframe)
    plt.xlabel("Experiments")
    plt.ylabel(metric_name)
    ax.set_xticklabels(metrics_list_names, rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(
        f"{figure_base_path}/boxplot/{metric_name}-{dataset}",
    )
    plt.close()


def test_significanse_student_t_test(series_1, series_2):
    res = stats.ttest_rel(series_1, series_2)
    return res[0], res[1]


def test_significanse(experiments: Dict[str, str], metric_name: str = "sMAPE"):
    metrics_list, metrics_list_names = metrics_experiment_lists(experiments, metric_name)
    # The first half is maped to the last half
    half_val = int(len(metrics_list_names)/2)
    print("Metric length", len(metrics_list_names))
    print("Half val", half_val)
    stat_values = []
    p_values = []
    exp_name = []
    for i in range(half_val):
        stat, p_value = test_significanse_student_t_test(
            metrics_list[i],
            metrics_list[(i+ half_val)]
        )
        stat_values.append(stat)
        p_values.append(p_value)
        name = "_".join(metrics_list_names[i].split(" ", 2)[:2])
        exp_name.append(
            name
        )
    return stat_values, p_values, exp_name

def test_significanse_multiple_datasets(experiments: List[Dict[str, str]], datasets: List[str], metric_name: str = "sMAPE", name="data", tabel_text="", table_save_path=table_save_path):
    dataset_stat_values = []
    dataset_p_values = []
    for exp in experiments:
        stat, p_value, exp_names = test_significanse(exp, metric_name)
        dataset_stat_values.append(
            stat
        )
        dataset_p_values.append(
            p_value
        )
    dataset_stat_values = DataFrame(dataset_stat_values, columns=exp_names)
    dataset_stat_values.index = datasets
    dataset_p_values = DataFrame(dataset_p_values, columns=exp_names)
    dataset_p_values.index = datasets

    utils.dataframe_to_latex_tabular(
        dataset_stat_values,
        f"{tabel_text} - stats",
        f"ttest-stats-{name}",
        add_index=True,
        save_local=table_save_path
    )
    utils.dataframe_to_latex_tabular(
        dataset_p_values,
        f"{tabel_text} - p-value",
        f"ttest-p-values-{name}",
        add_index=True,
        save_local=table_save_path
    )

