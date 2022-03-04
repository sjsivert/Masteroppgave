import logging
from collections import OrderedDict
from typing import Dict

import numpy as np
import torch
from torch import nn
from pipe import map, tee, where
from src.data_types.error_metrics_enum import ErrorMetricEnum
from src.utils.config_parser import config


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


def calculate_error(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Select the error defined first on the error list as the training error
    error_metric_selected = config["experiment"]["error_metrics"].get()[0]
    return choose_metric(try_convert_to_enum(error_metric_selected))(predictions, targets)


# TODO: Error here!
def calculate_errors(predictions: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
    error_metrics = config["experiment"]["error_metrics"].get()
    errors = OrderedDict(
        error_metrics
        | map(lambda key: try_convert_to_enum(key))
        | where(lambda metric: metric is not None)
        | map(lambda metric: (metric.value, choose_metric(metric)(predictions, targets)))
    )
    return errors


def calculate_mse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # nn.MSELoss()
    loss = torch.mean(torch.pow(targets - predictions, 2))
    return loss


def calculate_mae(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    loss = torch.mean((targets - predictions).abs())
    return loss


def calculate_mase(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # TODO!
    loss = torch.mean(targets)
    return loss


def calculate_smape(
    predictions: torch.Tensor, targets: torch.Tensor, batch: bool = True
) -> torch.Tensor:
    epsilon = 0.1
    loss = 2 * torch.mean(
        ((predictions - targets).abs()) / (predictions.abs() + targets.abs() + epsilon)
    )
    return loss
