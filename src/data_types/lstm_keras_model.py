import logging
from abc import ABC
from typing import Any, Dict, List, Optional, Union

import keras
import numpy as np
import optuna
import pipe
import tensorflow as tf
from numpy import ndarray
from pandas import DataFrame
from sklearn import metrics
from src.data_types.modules.lstm_keras_module import LstmKerasModule
from src.data_types.neural_net_keras_model import NeuralNetKerasModel
from src.data_types.neural_net_model import NeuralNetModel
from src.optuna_tuning.local_univariate_lstm_keras_objecktive import (
    local_univariate_lstm_keras_objective,
)
from src.pipelines import local_univariate_lstm_keras_pipeline as lstm_keras_pipeline
from src.pipelines import local_univariate_lstm_pipeline as lstm_pipeline
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from src.save_experiment_source.local_checkpoint_save_source import LocalCheckpointSaveSource
from src.utils.config_parser import config
from src.utils.keras_error_calculations import (
    config_metrics_to_keras_metrics,
    generate_error_metrics_dict,
)
from src.utils.keras_optimizer import KerasOptimizer
from src.utils.prettify_dict_string import prettify_dict_string
from tensorflow.keras.losses import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError


class LstmKerasModel(NeuralNetKerasModel, ABC):
    def __init__(
        self,
        log_sources: List[ILogTrainingSource],
        time_series_id: str,
        params: Dict,
        optuna_trial: Optional[optuna.trial.Trial] = None,
    ):
        super(LstmKerasModel, self).__init__(
            log_sources,
            time_series_id,
            params,
            optuna_trial,
        )
        self.should_shuffle_batches = params["should_shuffle_batches"]

    def init_neural_network(
        self, params: dict, logger=None, return_model: bool = False, **xargs
    ) -> Union[keras.Sequential, None]:
        # When tuning, update model parameters with the ones from the trial
        if not return_model:
            self.hyper_parameters = params

        model = LstmKerasModule(**params).model
        optim = KerasOptimizer.get(params["optimizer_name"], learning_rate=params["learning_rate"])

        keras_metrics = config_metrics_to_keras_metrics()
        model.compile(optimizer=optim, loss=keras_metrics[0], metrics=[keras_metrics])
        logging.info(
            f"Model compiled with optimizer {params['optimizer_name']}\n"
            f"{prettify_dict_string(params)} \
            \n{model.summary()}"
        )

        if return_model:
            return model
        else:
            self.model = model

    def train(self, epochs: int = None, **xargs) -> Dict:
        logging.info(f"Training {self.get_name()}")
        # TODO: Fix up this mess of repeated code. should only use dictionarys for hyperparameters
        self.batch_size = self.hyper_parameters["batch_size"]

        # This is commented out because we now have a fixed batch size and does not neeed to update datasets
        # self.split_data_sets()

        logging.info("Splitting training data into")

        is_tuning = xargs.pop("is_tuning") if "is_tuning" in xargs else False

        history = self.model.fit(
            x=self.x_train,
            y=self.y_train,
            epochs=self.hyper_parameters["number_of_epochs"],
            batch_size=self.batch_size,
            shuffle=self.should_shuffle_batches,
            validation_data=(self.x_val, self.y_val),
            **xargs,
        )
        history = history.history

        if not is_tuning:
            self._copy_trained_weights_to_model_with_different_batch_size()
            training_predictions, training_targets = self.predict_and_rescale(
                self.x_train, self.y_train
            )
            validation_predictions, validation_targets = self.predict_and_rescale(
                self.x_val, self.y_val.reshape(-1, 1)
            )
            self._visualize_predictions(
                (training_targets[:, 0].flatten()),
                (training_predictions[:, 0].flatten()),
                "Training predictions",
            )

            self._visualize_predictions(
                validation_targets.flatten(),
                validation_predictions.flatten(),
                "Validation predictions",
            )
            self._visualize_errors(
                [history["loss"], history["val_loss"]], ["Training_errors", "Validation_errors"]
            )

        self.metrics["training_error"] = history["loss"][-1]
        self.metrics["validation_error"] = history["val_loss"][-1]
        return self.metrics

    def _copy_trained_weights_to_model_with_different_batch_size(self) -> None:
        trained_weights = self.model.get_weights()
        params = self.hyper_parameters
        params["batch_size"] = 1
        prediction_model = self.init_neural_network(params=params, return_model=True)
        prediction_model.set_weights(trained_weights)
        self.prediction_model = prediction_model

    def predict_and_rescale(self, input_data: ndarray, targets: ndarray) -> ndarray:
        logging.info("Predicting")
        predictions = self.prediction_model.predict(input_data, batch_size=1)
        # predictions_rescaled = self._rescale_data(predictions)
        # targets_rescaled = self._rescale_data(targets)
        # predictions_rescaled = self._rescale_data(DataFrame(predictions))
        # targets_rescaled = self._rescale_data(DataFrame(predictions))

        # After fixing multivariate pipeline there was a bug that made rescaling not work
        # Therefore this is disabled for now
        predictions_rescaled = predictions
        targets_rescaled = targets

        return predictions_rescaled, targets_rescaled

    def _rescale_data(self, data: ndarray) -> ndarray:
        return self.min_max_scaler.inverse_transform(data)

    def test(self, predictive_period: int = 7, single_step: bool = False) -> Dict:
        logging.info("Testing")
        x_train, y_train = self.x_train, self.y_train
        x_test, y_test = self.x_test, self.y_test

        # Reset hidden states
        self.prediction_model.reset_states()
        results: List[float] = self.prediction_model.evaluate(
            x_train,
            y_train,
            batch_size=1,
        )
        results: List[float] = self.prediction_model.evaluate(
            x_test,
            y_test,
            batch_size=1,
        )
        # Remove first element because it it is a duplication of the second element.
        test_metrics = generate_error_metrics_dict(results[1:])

        # Visualize
        self.prediction_model.reset_states()
        self.predict_and_rescale(x_train, y_train)
        test_predictions, test_targets = self.predict_and_rescale(self.x_test, self.y_test)

        self._visualize_predictions(
            (test_targets.flatten()),
            (test_predictions.flatten()),
            "Test predictions",
        )
        self.metrics.update(test_metrics)
        return self.metrics

    def method_evaluation(
        self,
        parameters: Any,
        metric: str,
        singe_step: bool = True,
    ) -> Dict[str, Dict[str, str]]:
        logging.info("Tuning model")
        title, _ = LocalCheckpointSaveSource.load_title_and_description()
        study_name = f"{title}_{self.get_name()}"

        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(),
            # TODO: IModel should not rely on the config. Fix this
            storage=f"sqlite:///{config['experiment']['save_source']['disk']['model_save_location'].get()}/optuna-tuning.db"
            if len(self.log_sources) > 0
            else None,
            load_if_exists=True,
        )
        logging.info(
            f"Loading or creating optuna study with name: {study_name}\n"
            f"Number of previous Trials with this name are #{len(study.trials)}"
        )
        study.optimize(
            lambda trial: local_univariate_lstm_keras_objective(
                trial=trial,
                hyperparameter_tuning_range=parameters,
                model=self,
            ),
            timeout=parameters.get("time_to_tune_in_minutes", None),
            # TODO: Fix pytorch network to handle concurrency
            # n_jobs=-1,  # Use maximum number of cores
            n_trials=parameters.get("number_of_trials", None),
            # show_progress_bar=False,
            callbacks=[self.log_trial],
        )
        id = f"{self.get_name()},{study.best_trial.number}"
        best_params = study.best_trial.params
        logging.info("Best params!", best_params)
        test_params = self.hyper_parameters.copy()
        test_params.update(best_params)
        logging.info("Params updated with best params", test_params)
        self.init_neural_network(test_params)
        best_score = study.best_trial.value
        logging.info(
            f"Best trial: {id}\n" f"best_score: {best_score}\n" f"best_params: {best_params}"
        )
        self._generate_optuna_plots(study)

        return {id: {"best_score": best_score, "best_params": best_params}}

    def get_model(self):
        return self.model

    def save(self, path: str) -> str:
        save_path = f"{path}{self.get_name}.h5"
        self.model.save_weights(save_path)
        return save_path

    def load(self, path: str) -> None:
        load_path = f"{path}{self.get_name}.h5"
        self.model.load_weights(load_path)
