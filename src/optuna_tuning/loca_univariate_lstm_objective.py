from collections import OrderedDict
from typing import Tuple

import optuna

from src.data_types.lstm_model import LstmModel


def local_univariate_lstm_objective(
    trial: optuna.Trial, hyperparameter_tuning_range: OrderedDict[str, Tuple[int, int]]
) -> float:
    params = hyperparameter_range_to_optuna_range(trial, hyperparameter_tuning_range)
    model = LstmModel(**params)
    losses, val_losses = model.train_network(
        train_loader, val_loader, n_epochs=500, verbose=True, optuna_trial=trial
    )
    score = model.calculate_mean_score(val_losses)

    return score


def hyperparameter_range_to_optuna_range(
    trial: optuna.Trial, params: OrderedDict[str, Tuple[int, int]]
) -> OrderedDict[str, Tuple[float, float]]:
    params = {
        "number_of_features": params["number_of_features"],
        "hidden_size": trial.suggest_int(
            "hidden_size", params["hidden_size"][0], params["hidden_size"][1]
        ),
        "output_size": trial.suggest_int("output_size", params["output_size"]),
        "num_layers": trial.suggest_int(
            "num_layers", params["num_layers"][0], params["num_layers"][1]
        ),
        "learning_rate": trial.suggest_loguniform(
            "learning_rate", params["learning_rate"][0], params["learning_rate"][1]
        ),
        "batch_first": True,
        "dropout": 0.2,
        "bidirectional": False,
        # TODO: Find out how to change optimizer hyperparameters
        "optimizer_name": trial.suggest_categorical("optimizer_name", ["Adam", "RMSprop", "SGD"]),
    }
