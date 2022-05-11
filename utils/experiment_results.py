"""
Extract experiment results from model.
Create table with data from the ran models with parameters
1. One method for ARIMA
"""
from typing import Dict

import yaml
from pandas import DataFrame
import utils


experiment_path = "./models/"
experiment_name = "arima_experiment_dataset_2_MASE"
table_save_location = "./MastersThesis/tables/results"
table_name = "ARIMA dataset 2"
table_description = "experiment_arima_dataset2"


def config_extraction_arima():
    with open(f"{experiment_path}{experiment_name}/options.yaml") as f:
        config = yaml.full_load(f)
        params = config["model"]["local_univariate_arima"]["model_structure"]
    return params


# Extract ARIMA data
def arima_exp_results_table() -> Dict[str, Dict[str, float]]:
    # Get config from experiment
    metrics = {}
    with open(f"{experiment_path}{experiment_name}/metrics.txt") as f:
        content = f.readlines()
        current_model = None
        for line in content:
            line = line.strip("\n")
            if "Average" in line:  # No more data, average always last
                break
            elif "___" in line:  # A new model is used
                current_model = line.strip("_").split("-")[1]
                metrics[current_model] = {}
                continue
            elif current_model and line:
                metric_line = line.split(":")
                metric_value = float(metric_line[1].strip(" "))
                metric_name = metric_line[0].split("_")[1]
                metrics[current_model][metric_name] = metric_value
    return metrics


def arima_merge_metrics_and_params():
    config = config_extraction_arima()
    metrics = arima_exp_results_table()
    for model in config:
        dataset_id = model["time_series_id"]
        for param, val in model["hyperparameters"].items():
            metrics[str(dataset_id)][str(param)] = int(val)
    return metrics


def arima_dataframe() -> DataFrame:
    arima_metrics = arima_merge_metrics_and_params()
    metrics_data_frame = DataFrame(arima_metrics).transpose()
    return metrics_data_frame


if __name__ == "__main__":
    metrics = arima_dataframe()
    utils.dataframe_to_latex_tabular(
        metrics, table_name, table_description, add_index=True, save_local=table_save_location
    )
