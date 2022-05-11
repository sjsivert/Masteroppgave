from pandas import DataFrame
import utils
import numpy as np

# Table values
latex_caption = "Metrics for Dataset 1 Local Univariate LSTM model"
latex_label = "results:" + latex_caption.replace(" ", "_")
# Metrics average does not currenly work as expected. TODO Fix
metrics_average = False

# Select save location for generated table
table_save_path = "./MastersThesis/tables/results/"

# Select projects to be imported
base_path = "./models/"
# Multiple projects does not work?
projects = {
    # "arima": "dataset_1_arima",
    "local univariate lstm": "dataset_1-lstm-local-unviariate-tune-400-trials",
    # "local multivariate lstm": "dataset_1-lstm-multivariate-tune-400-trails",
    # "global univariate lstm": "dataset_1-lstm-global-unviariate-tune-400-trials",
    # "global multivariate lstm": "dataset_1-lstm-global-multivariate-tune-400-trials"
    # "lstm_dataset_1_local_univariate": "dataset_1-lstm-local-univariate-tune-400-trials",
    # "lstm": "dataset_1-lstm-global-unviariate-tune-400-trials",
    #    "lstm": "dataset_1-lstm-global-multivariate-tune-400-trials",
    #    "lstm": "dataset_2-lstm-global-univariate-tune-400-trials",
    #    "lstm": "dataset_2-lstm-local-univariate-tune-400-trials",
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
        updated_metrics["std"][metric_name] = round(np.std(average[metric_name]), 3)
        updated_metrics["mean"][metric_name] = round(np.mean(average[metric_name]), 3)
        updated_metrics["avg"][metric_name] = round(np.average(average[metric_name]), 3)
    return updated_metrics


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

    if metrics_average:
        scores = calc_average_values(scores)

    scores_list = []
    for key, value in scores.items():
        value["dataset"] = key
        scores_list.append(value)

    return scores_list


def fetch_metrics():
    for project_name, project_path in projects.items():
        metrics = extract_metrics_from_file(f"{base_path}{project_path}/metrics.txt")
        # metrics[project_name] = extract_metrics_from_file(
        #     f"{base_path}{project_path}/metrics.txt")
    metrics_data_frame = DataFrame(metrics)
    # Reorder pandas columns
    metrics_data_frame = change_pandas_colum_order(
        df=metrics_data_frame, col_name="dataset", index=0
    )

    return metrics_data_frame


def change_pandas_colum_order(df, col_name, index):
    col_names = df.columns.tolist()
    col_names.insert(index, col_names.pop(col_names.index(col_name)))
    df = df[col_names]
    return df


def export_latex_table():
    print("Fetching metrics..")
    print(projects)
    metrics_data = fetch_metrics()
    print(f"Generating latex tables in {table_save_path}")
    utils.dataframe_to_latex_tabular(
        metrics_data, latex_caption, latex_label, add_index=False, save_local=f"{table_save_path}"
    )


export_latex_table()
