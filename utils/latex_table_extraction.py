from pandas import DataFrame
import utils
import numpy as np

# Table values
table_name = "average_experiments"
table_description = "experiments"
metrics_average = False

# Select save location for generated table
table_save_path = "./MastersThesis/tables/results/"

# Select projects to be imported
base_path = "./models/"
projects = {
    "arima": "dataset_1_arima",
    "cnn-ae-lstm": "dataset_1_scale_test_lstm"
}

# Select metrics to be used
metric_types = ["mase", "smape", "days"]


def calc_average_values(metrics):
    average = {}
    for time_series, values in metrics.items():
        for metric_name, metric_value in values.items():
            if metric_name not in average:
                average[metric_name] = []
            average[metric_name].append(metric_value)
    # Calculate average, mean, std, ...
    updated_metrics = {}
    updated_metrics["std"] = {}
    updated_metrics["mean"] = {}
    updated_metrics["avg"] = {}
    for metric_name in average:
        updated_metrics["std"][metric_name] = round(np.std(average[metric_name]), 5)
        updated_metrics["mean"][metric_name] = round(np.mean(average[metric_name]), 5)
        updated_metrics["avg"][metric_name] = round(np.average(average[metric_name]), 5)
    return updated_metrics



    pass


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
            if metric_name not in metric_types:
                continue
            if "days" in metric_name:
                metric_name = metric_name_split.split("_", 1)[-1].replace("_", "-").lower()
            metric_value = float(metric_value)
            metrics[metric_name] = round(metric_value, 5)
        scores[data_set_name] = metrics

    if metrics_average:
        scores = calc_average_values(scores)

    return scores


def fetch_metrics():
    metrics = {}
    for project_name, project_path in projects.items():
        metrics[project_name] = extract_metrics_from_file(f"{base_path}{project_path}/metrics.txt")
    metrics_data_frame = DataFrame(metrics)
    return metrics_data_frame


def export_latex_table():
    metrics_data = fetch_metrics()
    utils.dataframe_to_latex_tabular(
        metrics_data,
        table_name,
        table_description,
        add_index=True,
        save_local=f"{table_save_path}"
    )


export_latex_table()
