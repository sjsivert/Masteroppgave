import logging
from abc import ABC
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from src.data_types.lstm_keras_model import LstmKerasModel
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from src.utils.keras_error_calculations import (
    config_metrics_to_keras_metrics,
    generate_error_metrics_dict,
    keras_mase_periodic,
)
from src.utils.lr_scheduler import scheduler
from tensorflow.keras.callbacks import LambdaCallback


class LstmKerasGlobalModel(LstmKerasModel, ABC):
    def __init__(
        self,
        log_sources: List[ILogTrainingSource],
        time_series_ids: List[int],
        params: Dict,
        optuna_trial: Optional[optuna.trial.Trial] = None,
    ):
        super(LstmKerasModel, self).__init__(
            log_sources,
            ",".join(str(time_series_id) for time_series_id in time_series_ids),
            params,
            optuna_trial,
        )
        self.time_series_ids = time_series_ids
        self.x_train_seperated = []
        self.y_train_seperated = []
        self.x_val_seperated = []
        self.y_val_seperated = []
        self.x_test_seperated = []
        self.y_test_seperated = []
        self.x_train_seperated_with_val_set = []
        self.y_train_seperated_with_val_set = []
        self.scalers = []

    def process_data(self, data_set: Any, training_size: float) -> None:
        x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []
        for cat_id in self.time_series_ids:
            data_pipeline = self.pipeline(
                data_set=data_set,
                cat_id=cat_id,
                input_window_size=self.input_window_size,
                output_window_size=self.output_window_size,
            )
            for log_source in self.log_sources:
                log_source.log_pipeline_steps(data_pipeline.__repr__())
            (
                training_data,
                testing_data,
                min_max_scaler,
                self.training_data_no_windows,
                self.training_data_without_diff,
            ) = data_pipeline.run()
            training_data_splitted, validation_data, testing_data = self.split_data_sets(
                training_data, testing_data
            )
            x_train.append(training_data_splitted[0])
            y_train.append(training_data_splitted[1])
            x_val.append(validation_data[0])
            y_val.append(validation_data[1])
            x_test.append(testing_data[0])
            y_test.append(testing_data[1])

            self.x_train_seperated.append(training_data_splitted[0])
            self.y_train_seperated.append(training_data_splitted[1])
            self.x_val_seperated.append(validation_data[0])
            self.y_val_seperated.append(validation_data[1])
            self.x_test_seperated.append(testing_data[0])
            self.y_test_seperated.append(testing_data[1])
            self.x_train_seperated_with_val_set.append(training_data[0])
            self.y_train_seperated_with_val_set.append(training_data[1])
            self.scalers.append(min_max_scaler)

        self.x_train = np.concatenate(x_train, axis=0)
        self.y_train = np.concatenate(y_train, axis=0)
        self.x_val = np.concatenate(x_val, axis=0)
        self.y_val = np.concatenate(y_val, axis=0)
        self.x_test = np.concatenate(x_test, axis=0)
        self.y_test = np.concatenate(y_test, axis=0)

        self.batch_count = 0
        self.time_series_count = 0
        self.batches_in_each_series = []
        for series in self.x_train_seperated:
            self.batches_in_each_series.append(series.shape[0] // self.batch_size)

    def reset_batch_counter(self):
        self.batch_count = 0
        self.time_series_count = 0

    def is_done_with_time_series(self, batch_number: int) -> bool:

        try:
            batch_number_grater_than_batches_in_series = (
                self.batch_count + self.batches_in_each_series[self.time_series_count]
                <= batch_number
            )
            if batch_number_grater_than_batches_in_series:
                self.batch_count += self.batches_in_each_series[self.time_series_count]
                self.time_series_count += 1
                return True
            return False
        except IndexError:
            return False

    def epoch_end_callback(self, epoch, logs):
        self.model.reset_states()
        self.reset_batch_counter()

    def train(self, epochs: int = None, **xargs) -> Dict:
        logging.info(f"Training {self.get_name()}")
        # TODO: Fix up this mess of repeated code. should only use dictionarys for hyperparameters
        self.batch_size = self.hyper_parameters["batch_size"]

        # This is commented out because we now have a fixed batch size and does not neeed to update datasets
        # self.split_data_sets()

        is_tuning = xargs.pop("is_tuning") if "is_tuning" in xargs else False

        if not is_tuning:
            x_train = np.concatenate([self.x_train, self.x_val], axis=0)
            y_train = np.concatenate([self.y_train, self.y_val], axis=0)
        else:
            x_train = self.x_train
            y_train = self.y_train

        reset_states_callback_on_time_series_end = LambdaCallback(
            on_batch_begin=lambda batch, logs: self.model.reset_states()
            if self.is_done_with_time_series(batch)
            else None
        )
        reset_states_on_epoch_begin_callback = LambdaCallback(
            on_epoch_begin=lambda epoch, logs: self.epoch_end_callback(epoch, logs)
        )
        reset_states_on_train_end = LambdaCallback(
            on_train_end=lambda logs: self.model.reset_states()
        )
        learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
        all_callbacks = [
            reset_states_callback_on_time_series_end,
            reset_states_on_epoch_begin_callback,
            reset_states_on_train_end,
            learning_rate_callback,
        ]
        all_callbacks.extend(xargs.pop("callbacks", []))

        history = self.model.fit(
            x=x_train,
            y=y_train,
            epochs=self.hyper_parameters["number_of_epochs"],
            batch_size=self.batch_size,
            shuffle=self.should_shuffle_batches,
            validation_data=(self.x_val, self.y_val),
            callbacks=all_callbacks,
        )
        history = history.history

        if not is_tuning:
            self._copy_trained_weights_to_model_with_different_batch_size()

            # Cannot rescale data because it consists of multiple datasets with multiple scalers
            training_predictions = self.prediction_model.predict(x_train, batch_size=1)
            training_targets = y_train

            # validation_predictions, validation_targets = self.predict_and_rescale(
            #     self.x_val, self.y_val.reshape(-1, 1)
            # )
            validation_predictions = self.prediction_model.predict(self.x_val, batch_size=1)
            validation_targets = self.y_val
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

    def test(self, predictive_period: int = 7, single_step: bool = False) -> Dict:
        logging.info("Testing")
        global_metrics = {}

        for i in range(len(self.x_test_seperated)):
            testing_set_name = self.time_series_ids[i]
            self.name = testing_set_name
            x_test = self.x_test_seperated[i]
            y_test = self.y_test_seperated[i]
            self.min_max_scaler = self.scalers[i]
            x_train = np.concatenate(
                [self.x_train_seperated_with_val_set[i], self.x_val_seperated[i]], axis=0
            )
            y_train = np.concatenate(
                [self.y_train_seperated_with_val_set[i], self.y_val_seperated[i]], axis=0
            )

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
            # Copy LSTM weights for different batch sizes
            self.prediction_model.reset_states()
            self.prediction_model.predict(x_train, batch_size=1)
            test_predictions = self.prediction_model.predict(x_test, batch_size=1)

            # Visualize, measure metrics
            self._lstm_test_scale_predictions(
                x_train,
                y_train,
                x_test,
                y_test,
                test_metrics,
                test_predictions,
                self.prediction_model,
            )

            # Create global metrics, one for each series
            self.metrics.update(test_metrics)
            global_metrics[testing_set_name] = self.metrics.copy()
            self.metrics = {}

        self.metrics = global_metrics
        # Run predictions on all data as well!
        # super().test()
        return self.metrics

    def predict_and_rescale(
        self, input_data: np.ndarray, targets: np.ndarray, scaler: MinMaxScaler = None
    ) -> np.ndarray:
        logging.info("Predicting")
        predictions = self.prediction_model.predict(input_data, batch_size=1)
        predictions_rescaled = scaler.inverse_transform(predictions) if scaler else predictions
        targets_rescaled = scaler.inverse_transform(targets) if scaler else targets

        return predictions_rescaled, targets_rescaled
