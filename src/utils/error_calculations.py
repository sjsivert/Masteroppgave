import logging
import random
from collections import OrderedDict
from statistics import mean
from typing import Dict
from xml.dom import NotFoundErr

import numpy as np
from pandas import DataFrame
from pipe import map, tee, where
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from packages.permetrics.permetrics.regression import Metrics
from src.data_types.error_metrics_enum import ErrorMetricEnum
from src.utils.config_parser import config


def choose_metric(metric: ErrorMetricEnum):
    if metric == ErrorMetricEnum.MSE:
        return calculate_mse
    elif metric == ErrorMetricEnum.MAE:
        return calculate_mae
    elif metric == ErrorMetricEnum.RMSE:
        return calculate_rmse
    elif metric == ErrorMetricEnum.MAPE:
        return calculate_mape
    elif metric == ErrorMetricEnum.MASE:
        return calculate_mase
    elif metric == ErrorMetricEnum.SMAPE:
        return calculate_smape
    elif metric == ErrorMetricEnum.OWA:
        return calculate_owa


def calculate_mse(data_set: DataFrame, proposed_data_set: DataFrame) -> float:
    mse = mean_squared_error(data_set, proposed_data_set)
    return mse


def calculate_mae(data_set: DataFrame, proposed_data_set: DataFrame) -> float:
    mae = mean_absolute_error(data_set, proposed_data_set)
    return mae


def calculate_rmse(data_set: DataFrame, proposed_data_set: DataFrame) -> float:
    return np.sqrt(calculate_mse(data_set, proposed_data_set))


def calculate_mape(data_set: DataFrame, proposed_data_set: DataFrame) -> float:
    mape = mean_absolute_percentage_error(data_set, proposed_data_set)
    return mape


# Error metrics
def calculate_mase(data_set: DataFrame, proposed_data_set: DataFrame) -> float:
    # Convert data_sets to numpy arrays
    metric = Metrics(data_set.to_numpy(), proposed_data_set.to_numpy())
    err = metric.MASE()
    return float(err)


def calculate_smape(data_set: DataFrame, proposed_data_set: DataFrame) -> float:
    metric = Metrics(data_set.to_numpy(), proposed_data_set.to_numpy())
    err = metric.SMAPE()
    return float(err)


def calculate_owa(data_set: DataFrame, proposed_data_set: DataFrame) -> float:
    metric = Metrics(data_set.to_numpy(), proposed_data_set.to_numpy())
    err = metric.OWA()
    return float(err)


def try_convert_to_enum(key: str) -> ErrorMetricEnum:
    try:
        return ErrorMetricEnum[key]
    except KeyError:
        logging.warning(
            f" '{key}' is not an implemented error metric. Valid values are {ErrorMetricEnum.__members__}"
        )


def calculate_error(data_set: DataFrame, propsed_data_set: DataFrame) -> Dict[str, float]:
    error_metrics = config["experiment"]["error_metrics"].get()
    errors = OrderedDict(
        error_metrics
        | map(lambda key: try_convert_to_enum(key))
        | where(lambda metric: metric is not None)
        | map(lambda metric: (metric.value, choose_metric(metric)(data_set, propsed_data_set)))
    )
    return errors


if __name__ == "__main__":
    mock_dataset = DataFrame([random.randint(1, 40) for _ in range(50)])
    prediction = DataFrame([random.randint(1, 40) for _ in range(50)])
    calculate_error(mock_dataset, prediction)
