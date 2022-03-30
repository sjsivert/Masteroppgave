import logging
from collections import OrderedDict
from typing import Dict

import numpy as np
import torch
from pipe import map, tee, where
from src.data_types.error_metrics_enum import ErrorMetricEnum
from src.utils.config_parser import config
from torch import nn

# TODO!
"""
Is error calculated correctly over a batch?
TODO! This should be verified, as this could cause wrong error calculations
"""


def choose_metric(metric: ErrorMetricEnum):
    if metric == ErrorMetricEnum.MSE:
        return calculate_mse
    elif metric == ErrorMetricEnum.MAE:
        return calculate_mae
    elif metric == ErrorMetricEnum.MASE:
        return calculate_mase
    elif metric == ErrorMetricEnum.SMAPE:
        return calculate_smape


def try_convert_to_enum(key: str) -> ErrorMetricEnum:
    try:
        return ErrorMetricEnum[key]
    except KeyError:
        logging.warning(
            f" '{key}' is not an implemented error metric. Valid values are {ErrorMetricEnum.__members__}"
        )


def calculate_error(targets: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    # Select the error defined first on the error list as the training error
    error_metric_selected = config["experiment"]["error_metrics"].get()[0]
    return choose_metric(try_convert_to_enum(error_metric_selected))(targets, predictions)


# TODO: Error here!
def calculate_errors(targets: torch.Tensor, predictions: torch.Tensor) -> Dict[str, float]:
    error_metrics = config["experiment"]["error_metrics"].get()
    errors = OrderedDict(
        error_metrics
        | map(lambda key: try_convert_to_enum(key))
        | where(lambda metric: metric is not None)
        | map(lambda metric: (metric.value, choose_metric(metric)(targets, predictions).item()))
    )
    return errors


def calculate_mse(targets: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    # nn.MSELoss()
    loss = torch.mean(torch.pow(targets - predictions, 2))
    return loss


def calculate_mae(targets: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    loss = torch.mean((targets - predictions).abs())
    return loss


def calculate_mase(targets: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    # Calculate mae for predictions and targets, and for naive predictions
    naive = targets.clone()
    # Assume the first dim is the batch and a 3 dim tensor
    naive[:, 1:] = naive[:, :-1].clone()
    naive[1:, 0, 0] = naive[:-1, 0, 0].clone()
    loss_1 = calculate_mae(targets, predictions)
    loss_2 = calculate_mae(targets, naive)
    loss = loss_1 / loss_2
    return loss


def calculate_smape(targets: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    epsilon = 0.1
    loss = 2 * torch.mean(((predictions - targets).abs()) / (predictions.abs() + targets.abs()))
    return loss
