from typing import Dict
from pandas import DataFrame
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_mse(data_set: DataFrame, proposed_data_set: DataFrame) -> float:
    mse = mean_squared_error(data_set, proposed_data_set)
    return mse


def calculate_mae(data_set: DataFrame, proposed_data_set: DataFrame) -> float:
    mae = mean_absolute_error(data_set, proposed_data_set)
    return mae


# TODO: Add additional error metrics


def calculate_error(data_set: DataFrame, propsed_data_set: DataFrame) -> Dict[str, float]:
    errors = {}
    errors["MSE"] = calculate_mse(data_set, propsed_data_set)
    errors["MAE"] = calculate_mae(data_set, propsed_data_set)
    return errors
