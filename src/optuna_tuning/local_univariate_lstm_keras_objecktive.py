import logging
import typing
from collections import OrderedDict
from typing import Dict, List, OrderedDict, Tuple

import keras.backend as K
import optuna
from optuna.integration import (
    KerasPruningCallback,
    PyTorchLightningPruningCallback,
    TFKerasPruningCallback,
    pytorch_lightning,
)
from src.data_types.i_model import IModel
from src.data_types.modules.lstm_module import LstmModule
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from src.utils.prettify_dict_string import prettify_dict_string
from src.utils.time_function import time_function
from torch.utils.data import DataLoader


def local_univariate_lstm_keras_objective(
    trial: optuna.Trial,
    hyperparameter_tuning_range: OrderedDict[str, Tuple[int, int]],
    model: IModel,
) -> float:

    # Clear keras session to avoid memory leak
    K.clear_session()

    params = hyperparameter_range_to_optuna_range(trial, hyperparameter_tuning_range)

    # The default logger in PyTorch Lightning writes to event files to be consumed by
    # TensorBoard. We create a simple logger instead that holds the log in memory so that the
    # final accuracy can be obtained after optimization. When using the default logger, the
    # final accuracy could be stored in an attribute of the `Trainer` instead.

    logging.info(
        f"Starting tuning trial number #{trial.number} of total {hyperparameter_tuning_range['number_of_trials']}\n"
    )

    model.init_neural_network(
        params=params,
        return_model=False,
        # TODO: Fix for pruning for keras
        # callbacks=[PyTorchLightningPruningCallback(trial, monitor="validation_loss")],
    )

    with time_function():
        errors = model.train(
            epochs=params["number_of_epochs"],
            is_tuning=True,
            callbacks=[TFKerasPruningCallback(trial, "val_loss")],
        )
        # TODO: Use config parameter 'metric'to use when tuning
        # score = model.calculate_mean_score(errors[""])

    # return errors["training_error"]
    return errors["validation_error"]


def local_univariate_cnn_ae_keras_objective(
    trial: optuna.Trial,
    hyperparameter_tuning_range: OrderedDict[str, Tuple[int, int]],
    model: IModel,
) -> float:
    # Clear keras session to avoid memory leak
    K.clear_session()

    params = hyperparameter_range_to_optuna_range_cnn(trial, hyperparameter_tuning_range)

    # The default logger in PyTorch Lightning writes to event files to be consumed by
    # TensorBoard. We create a simple logger instead that holds the log in memory so that the
    # final accuracy can be obtained after optimization. When using the default logger, the
    # final accuracy could be stored in an attribute of the `Trainer` instead.

    logging.info(
        f"Starting tuning trial number #{trial.number} of total {hyperparameter_tuning_range['number_of_trials']}\n"
    )

    model.init_neural_network(
        params=params,
        return_model=False,
        # TODO: Fix for pruning for keras
        # callbacks=[PyTorchLightningPruningCallback(trial, monitor="validation_loss")],
    )

    with time_function():
        errors, _ = model.train_lstm(
            epochs=params["number_of_epochs"],
            callbacks=[TFKerasPruningCallback(trial, "val_loss")],
            tuning=True,
        )
        # TODO: Use config parameter 'metric'to use when tuning
        # score = model.calculate_mean_score(errors[""])
    return errors["validation_error"]


def local_univariate_cnn_ae_lstm_keras_objective(
    trial: optuna.Trial,
    hyperparameter_tuning_range: OrderedDict[str, Tuple[int, int]],
    model: IModel,
) -> float:

    # Clear keras session to avoid memory leak
    K.clear_session()

    params = hyperparameter_range_to_optuna_range(trial, hyperparameter_tuning_range)

    # The default logger in PyTorch Lightning writes to event files to be consumed by
    # TensorBoard. We create a simple logger instead that holds the log in memory so that the
    # final accuracy can be obtained after optimization. When using the default logger, the
    # final accuracy could be stored in an attribute of the `Trainer` instead.

    logging.info(
        f"Starting tuning trial number #{trial.number} of total {hyperparameter_tuning_range['number_of_trials']}\n"
    )

    model.init_autoencoder_and_lstm(
        params=params,
        return_model=False,
        # TODO: Fix for pruning for keras
        # callbacks=[PyTorchLightningPruningCallback(trial, monitor="validation_loss")],
    )

    with time_function():
        errors, _ = model.train_lstm(
            epochs=params["number_of_epochs"],
            callbacks=[TFKerasPruningCallback(trial, "val_loss")],
            tuning=True,
        )
        # TODO: Use config parameter 'metric'to use when tuning
        # score = model.calculate_mean_score(errors[""])
    return errors["validation_error"]


def hyperparameter_range_to_optuna_range_cnn(
    trial: optuna.Trial, config_params: OrderedDict[str, Tuple[int, int]]
) -> Dict[str, Tuple[float, float]]:
    pass


def hyperparameter_range_to_optuna_range(
    trial: optuna.Trial, config_params: OrderedDict[str, Tuple[int, int]]
) -> Dict[str, Tuple[float, float]]:
    number_of_layers = (
        trial.suggest_int(
            "number_of_layers",
            config_params["number_of_layers"][0],
            config_params["number_of_layers"][1],
        ),
    )
    layers = []
    for layer in range(number_of_layers[0]):
        layer_params = {
            "dropout": trial.suggest_float(
                f"dropout_{layer}", config_params["dropout"][0], config_params["dropout"][1]
            ),
            "hidden_size": trial.suggest_int(
                f"hidden_size_{layer}",
                config_params["hidden_size"][0],
                config_params["hidden_size"][1],
            ),
            "recurrent_dropout": trial.suggest_float(
                f"recurrent_dropout_{layer}",
                config_params["recurrent_dropout"][0],
                config_params["recurrent_dropout"][1],
            ),
        }
        layers.append(layer_params)
    return {
        "number_of_features": config_params["number_of_features"],
        # "hidden_layer_size": trial.suggest_int(
        #     "hidden_layer_size", config_params["hidden_size"][0], config_params["hidden_size"][1]
        # ),
        "input_window_size": config_params["input_window_size"],
        "output_window_size": config_params["output_window_size"],
        # "number_of_layers": trial.suggest_int(
        #     "number_of_layers",
        #     config_params["number_of_layers"][0],
        #     config_params["number_of_layers"][1],
        # ),
        "learning_rate": trial.suggest_loguniform(
            "learning_rate",
            float(config_params["learning_rate"][0]),
            float(config_params["learning_rate"][1]),
        ),
        "layers": layers,
        "batch_first": True,
        # "batch_size": config_params["batch_size"],
        "batch_size": trial.suggest_int(
            "batch_size", config_params["batch_size"][0], config_params["batch_size"][1]
        ),
        # "dropout": trial.suggest_float(
        #     "dropout", config_params["dropout"][0], config_params["dropout"][1]
        # ),
        "bidirectional": False,
        # TODO: Find out how to change optimizer hyperparameters
        "optimizer_name": trial.suggest_categorical(
            "optimizer_name", config_params["optimizer_name"]
        ),
        "number_of_epochs": trial.suggest_int(
            "number_of_epochs",
            config_params["number_of_epochs"][0],
            config_params["number_of_epochs"][1],
        ),
        "stateful_lstm": config_params["stateful_lstm"],
    }
