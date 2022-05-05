import logging
from abc import ABC
from subprocess import call
from tabnanny import verbose
from typing import Any, Dict, List, Optional, Union

import keras
import numpy as np
import optuna
import pipe
import tensorflow as tf
from numpy import ndarray
from pandas import DataFrame
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from src.data_types.modules.lstm_keras_module import LstmKerasModule
from src.data_types.neural_net_keras_model import NeuralNetKerasModel
from src.data_types.neural_net_model import NeuralNetModel
from src.optuna_tuning.local_univariate_lstm_keras_objecktive import \
    local_univariate_lstm_keras_objective
from src.pipelines import \
    local_univariate_lstm_keras_pipeline as lstm_keras_pipeline
from src.pipelines import local_univariate_lstm_pipeline as lstm_pipeline
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from src.save_experiment_source.local_checkpoint_save_source import \
    LocalCheckpointSaveSource
from src.utils.config_parser import config, update_config_lstm_params
from src.utils.keras_error_calculations import (
    config_metrics_to_keras_metrics, generate_error_metrics_dict, keras_mase,
    keras_mase_periodic)
from src.utils.keras_optimizer import KerasOptimizer
from src.utils.lr_scheduler import scheduler
from src.utils.prettify_dict_string import prettify_dict_string
from src.utils.reverse_pipeline import (
    reverse_decrease_variance,
    reverse_differencing,
    reverse_differencing_forecast,
    reverse_sliding_window,
)
from src.utils.visuals import visualize_data_series
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.losses import (MeanAbsoluteError,
                                     MeanAbsolutePercentageError,
                                     MeanSquaredError)
from zmq import PROTOCOL_ERROR_ZMTP_MALFORMED_COMMAND_UNSPECIFIED


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

        model = LstmKerasModule(**params).model
        optim = KerasOptimizer.get(params["optimizer_name"], learning_rate=params["learning_rate"])

        keras_metrics = config_metrics_to_keras_metrics()
        model.compile(optimizer=optim, loss=keras_metrics[0], metrics=[keras_metrics])
        round(model.optimizer.lr.numpy(), 5)
        logging.info(
            f"Model compiled with optimizer {params['optimizer_name']}\n"
            f"{prettify_dict_string(params)} \
            \n{model.summary()}"
        )

        copy_of_params = params.copy()
        copy_of_params.pop("batch_size")
        self.hyper_parameters.update(copy_of_params)
        print(self.hyper_parameters)
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

        if not is_tuning:
            x_train = np.concatenate([self.x_train, self.x_val], axis=0)
            y_train = np.concatenate([self.y_train, self.y_val], axis=0)
        else:
            x_train = self.x_train
            y_train = self.y_train
        callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
        callbacks = [callback] + xargs.pop("callbacks", [])
        history = self.model.fit(
            x=x_train,
            y=y_train,
            epochs=self.hyper_parameters["number_of_epochs"],
            batch_size=self.batch_size,
            shuffle=self.should_shuffle_batches,
            validation_data=(self.x_val, self.y_val),
            callbacks=[callbacks],
            **xargs,
        )
        history = history.history

        if not is_tuning:
            self._copy_trained_weights_to_model_with_different_batch_size()
            # training_predictions, training_targets = self.predict_and_rescale(x_train, y_train)
            training_predictions = self.prediction_model.predict(x_train, batch_size=1)
            validation_predictions = self.prediction_model.predict(self.x_val, batch_size=1)

            training_data_original_scale = self._reverse_pipeline_training(
                self.training_data_no_windows, self.training_data_without_diff
            )
            training_predictions_original_scale = self._reverse_pipeline_training(
                training_predictions[:, 0], self.training_data_without_diff
            )
            self._visualize_predictions(
                (training_data_original_scale),
                (training_predictions_original_scale),
                "Training predictions original scale",
            )
            self._visualize_predictions(
                (self.training_data_no_windows),
                (training_predictions.flatten()),
                "Training predictions",
            )
            self._visualize_predictions(
                (self.x_train[:, 0, 0].flatten()),
                (self.y_train[0, 0].flatten()),
                "Training predictions without validation set",
            )

            self._visualize_predictions(
                self.y_val.flatten(),
                validation_predictions.flatten()
                if validation_predictions.shape[0] > 1
                else validation_predictions[:, 0].flatten(),
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
        predictions_rescaled = self._rescale_data(predictions)
        targets_rescaled = self._rescale_data(targets)
        # predictions_rescaled = self._rescale_data(DataFrame(predictions))
        # targets_rescaled = self._rescale_data(DataFrame(predictions))

        # After fixing multivariate pipeline there was a bug that made rescaling not work
        # Therefore this is disabled for now
        # predictions_rescaled = predictions
        # targets_rescaled = targets

        return predictions_rescaled, targets_rescaled

    def _rescale_data(self, data: ndarray) -> ndarray:
        return self.min_max_scaler.inverse_transform(data) if self.min_max_scaler else data

    def test(self, predictive_period: int = 7, single_step: bool = False) -> Dict:
        logging.info("Testing")
        x_train = np.concatenate([self.x_train, self.x_val], axis=0)
        y_train = np.concatenate([self.y_train, self.y_val], axis=0)
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
        # Remove first element because it is a duplication of the second element.
        test_metrics = generate_error_metrics_dict(results[1:])

        # Visualize
        self.prediction_model.reset_states()
        # self.predict_and_rescale(x_train, y_train)
        self.prediction_model.predict(x_train, batch_size=1)
        test_predictions = self.prediction_model.predict(self.x_test, batch_size=1)

        test_predictions_reversed, test_targets_reversted = self.predict_and_reverse_pipeline(
            self.x_test, self.y_test, self.min_max_scaler, self.training_data_without_diff
        )

        # last_period_targets = (
        #     self.min_max_scaler.inverse_transform(x_test[:, -self.output_window_size:])
        #     if self.min_max_scaler
        #     else x_test[:, -self.output_window_size:]
        # )
        last_period_targets = self._reverse_pipeline_training(
            training_data=self.x_test[:, -self.output_window_size :],
            original_data=self.training_data_without_diff[-self.output_window_size :, :],
        )

        mase_seven_days, y_true_last_period = keras_mase_periodic(
            y_true=self.y_test,
            y_true_last_period=last_period_targets,
            y_pred=test_predictions_reversed,
        )

        # Custom evaluate function with rescale before metrics
        self.prediction_model.reset_states()
        self.prediction_model.predict(x_train, batch_size=1)
        custom_metrics, _ = self.custom_evaluate(
            x_test=x_test,
            y_test=y_test,
        )
        self.metrics.update(custom_metrics)
        print("CUSTOM METRRICS-----------")
        print(custom_metrics)
        # self.metrics.update(custom_metrics)
        test_metrics[f"test_MASE_{len(self.x_test)}_DAYS"] = mase_periode.numpy()
        self.metrics.update(custom_metrics)

        self._visualize_predictions(
            (self.y_test.flatten()),
            (test_predictions_reversed.flatten()),
            "Test predictions",
        )
        self._visualize_predictions_and_last_period(
            (self.y_test.flatten()),
            (test_predictions_reversed.flatten()),
            last_period_targets.flatten(),
            "Test predictions with last period targets",
        )
        x_test_values = self._reverse_pipeline_training(
            training_data=x_test[:, -self.output_window_size :],
            original_data=self.training_data_without_diff[-self.output_window_size :],
        )
        self._visualize_predictions_with_context(
            context=x_test_values.flatten(),
            targets=self.y_test.flatten(),
            predictions=test_predictions_reversed.flatten(),
        )
        self.metrics.update(test_metrics)
        training_data_original_scale = self._reverse_pipeline_training(
            self.training_data_no_windows, self.training_data_without_diff
        )
        return self.metrics

    def predict_and_reverse_pipeline(
        self, input_data: ndarray, targets: ndarray, min_max_scaler: StandardScaler, original_data
    ) -> ndarray:
        # reverse training data
        logging.info("Predicting")
        predictions = self.prediction_model.predict(input_data, batch_size=1)
        predictions_reversed_pipeline = self._reverse_pipeline(
            predictions, min_max_scaler, original_data
        )
        return predictions_reversed_pipeline, targets

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
        best_params["time_series_id"] = self.get_name()
        logging.info("Best params!", best_params)
        params_copied = self.hyper_parameters.copy()
        params_copied.update(best_params)
        logging.info("Params updated with best params", params_copied)
        self.init_neural_network(params_copied)
        best_score = study.best_trial.value
        logging.info(
            f"Best trial: {id}\n" f"best_score: {best_score}\n" f"best_params: {best_params}"
        )
        self._generate_optuna_plots(study)

        # Update config with best params
        update_config_lstm_params(best_params)

        return {id: {"best_score": best_score, "best_params": best_params}}

    def get_model(self):
        return self.model

    def save(self, path: str) -> str:
        save_path = f"{path}{self.get_name()}.h5"
        self.model.save_weights(save_path)
        return save_path

    def load(self, path: str) -> None:
        load_path = f"{path}{self.get_name()}.h5"
        self.model.load_weights(load_path)

    def custom_evaluate(self, x_test, y_test, post_processing=None):
        predictions = self.prediction_model.predict(x_test, batch_size=1)

        predictions_re_composed = self._reverse_pipeline(
            predictions, self.min_max_scaler, self.training_data_without_diff
        )
        # TODO Post processing
        results = {}
        kerast_metrics_to_calculate = [
            tf.keras.metrics.MeanSquaredError,
            tf.keras.metrics.MeanAbsoluteError,
            tf.keras.metrics.MeanAbsolutePercentageError,
        ]

        for metric_func in kerast_metrics_to_calculate:
            metric = metric_func()
            metric.update_state(y_test, predictions_re_composed)
            results[metric.name] = metric.result().numpy()
        results["mase"] = keras_mase(y_true=y_test, y_pred=predictions_re_composed).numpy()
        # results["smape"] = keras_smape(y_true=y_test, y_pred=predictions.numpy()).numpy()
        return results, predictions

    def _reverse_pipeline(self, predictions, min_max_scaler, original_data):
        predictions_scaled = self._rescale_data(predictions)
        predictions_reversed_diff = reverse_differencing_forecast(
            noise=predictions_scaled, last_observed=original_data[-1]
        )
        predictions_added_variance = reverse_decrease_variance(predictions_reversed_diff)

        return predictions_added_variance

    def _reverse_pipeline_training(self, training_data, original_data):
        # visualize_data_series(
        #     title=f"original data",
        #     data_series=[training_data],
        #     data_labels=["Targets", ],
        #     colors=["blue",],
        #     x_label="Time",
        #     y_label="Interest",
        # ).savefig("original_training_data_before_post_prosessing.png")

        rescaled = self._rescale_data(training_data.reshape(1, -1))
        # visualize_data_series(
        #     title=f"rescaled data",
        #     data_series=[rescaled],
        #     data_labels=["Targets", ],
        #     colors=["blue",],
        #     x_label="Time",
        #     y_label="Interest",
        # ).savefig("rescaled_data.png")

        reverse_diff = reverse_differencing(rescaled, original_data)

        # visualize_data_series(
        #     title=f"re-reverse_diff",
        #     data_series=[reverse_diff],
        #     data_labels=["Targets", ],
        #     colors=["blue",],
        #     x_label="Time",
        #     y_label="Interest",
        # ).savefig("re-reverse-diff.png")
        increase_variance = reverse_decrease_variance(reverse_diff)
        # visualize_data_series(
        #     title=f"incverage_variance",
        #     data_series=[increase_variance],
        #     data_labels=["Targets", ],
        #     colors=["blue",],
        #     x_label="Time",
        #     y_label="Interest",
        # ).savefig("increase_variances.png")
        return increase_variance
