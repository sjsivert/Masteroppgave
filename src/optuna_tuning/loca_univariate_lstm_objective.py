from collections import OrderedDict
from typing import Tuple

import optuna
from torch.utils.data import DataLoader

from src.data_types.lstm_model import LstmModel
from typing import OrderedDict


def local_univariate_lstm_objective(
    trial: optuna.Trial,
    hyperparameter_tuning_range: OrderedDict[str, Tuple[int, int]],
    train_data_loader: DataLoader,
    validation_data_loader: DataLoader,
    number_of_epoochs: int,
) -> float:
    params = hyperparameter_range_to_optuna_range(trial, hyperparameter_tuning_range)
    model = LstmModel(**params)
    losses, val_losses = model.train_network(
        train_data_loader,
        validation_data_loader,
        n_epochs=number_of_epoochs,
        verbose=True,
        optuna_trial=trial,
    )
    score = model.calculate_mean_score(val_losses)

    return score


def hyperparameter_range_to_optuna_range(
    trial: optuna.Trial, config_params: OrderedDict[str, Tuple[int, int]]
) -> OrderedDict[str, Tuple[float, float]]:
    return OrderedDict(
        {
            "number_of_features": config_params["number_of_features"],
            "hidden_size": trial.suggest_int(
                "hidden_size", config_params["hidden_size"][0], config_params["hidden_size"][1]
            ),
            "output_size": trial.suggest_int("output_size", config_params["output_size"]),
            "num_layers": trial.suggest_int(
                "num_layers", config_params["num_layers"][0], config_params["num_layers"][1]
            ),
            "learning_rate": trial.suggest_loguniform(
                "learning_rate",
                config_params["learning_rate"][0],
                config_params["learning_rate"][1],
            ),
            "batch_first": True,
            "dropout": 0.2,
            "bidirectional": False,
            # TODO: Find out how to change optimizer hyperparameters
            "optimizer_name": trial.suggest_categorical(
                "optimizer_name", config_params["optimizer_name"]
            ),
        }
    )
