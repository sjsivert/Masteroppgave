import typing
from collections import OrderedDict
from typing import Tuple, Dict, List

import optuna
from torch.utils.data import DataLoader

from typing import OrderedDict

from src.data_types.i_model import IModel
from src.data_types.modules.lstm_module import LstmModule
from src.save_experiment_source.i_log_training_source import ILogTrainingSource


def local_univariate_lstm_objective(
    trial: optuna.Trial,
    hyperparameter_tuning_range: OrderedDict[str, Tuple[int, int]],
    metric_to_use_when_tuning: str,
    model: IModel,
) -> float:
    params = hyperparameter_range_to_optuna_range(trial, hyperparameter_tuning_range)

    number_of_epochs = (
        trial.suggest_int(
            "number_of_epochs",
            hyperparameter_tuning_range["number_of_epochs"][0],
            hyperparameter_tuning_range["number_of_epochs"][1],
        ),
    )

    model.init_neural_network()
    errors = model.train(
        epochs=number_of_epochs[0],
    )
    # TODO: Use config parameter 'metric'to use when tuning
    score = model.calculate_mean_score(errors["validation_error"])
    print("score:", score)

    return score


def hyperparameter_range_to_optuna_range(
    trial: optuna.Trial, config_params: OrderedDict[str, Tuple[int, int]]
) -> Dict[str, Tuple[float, float]]:
    return {
        "number_of_features": config_params["number_of_features"],
        "hidden_layer_size": trial.suggest_int(
            "hidden_layer_size", config_params["hidden_size"][0], config_params["hidden_size"][1]
        ),
        "output_window_size": config_params["output_window_size"],
        "number_of_layers": trial.suggest_int(
            "number_of_layers",
            config_params["number_of_layers"][0],
            config_params["number_of_layers"][1],
        ),
        "learning_rate": trial.suggest_loguniform(
            "learning_rate",
            float(config_params["learning_rate"][0]),
            float(config_params["learning_rate"][1]),
        ),
        "batch_first": True,
        "dropout": 0.2,
        "bidirectional": False,
        # TODO: Find out how to change optimizer hyperparameters
        "optimizer_name": trial.suggest_categorical(
            "optimizer_name", config_params["optimizer_name"]
        ),
    }
