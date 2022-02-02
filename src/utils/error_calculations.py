import logging
import random
from typing import Dict
from xml.dom import NotFoundErr

from pandas import DataFrame
from pipe import map, tee, where
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data_types.error_metrics_enum import ErrorMetricEnum
from src.utils.config_parser import config


def choose_metric(metric: ErrorMetricEnum):
    if metric == ErrorMetricEnum.MSE:
        return calculate_mse
    elif metric == ErrorMetricEnum.MAE:
        return calculate_mae


def calculate_mse(data_set: DataFrame, proposed_data_set: DataFrame) -> float:
    mse = mean_squared_error(data_set, proposed_data_set)
    return mse


def calculate_mae(data_set: DataFrame, proposed_data_set: DataFrame) -> float:
    mae = mean_absolute_error(data_set, proposed_data_set)
    return mae


# TODO: Add additional error metrics


def try_convert_to_enum(key: str) -> ErrorMetricEnum:
    try:
        return ErrorMetricEnum[key]
    except KeyError:
        logging.warning(
            f" '{key}' is not an implemented error metric. Valid values are {ErrorMetricEnum.__members__}"
        )


def calculate_error(data_set: DataFrame, propsed_data_set: DataFrame) -> Dict[str, float]:
    error_metrics = config["experiment"]["error_metrics"].get()

    errors = dict(
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
