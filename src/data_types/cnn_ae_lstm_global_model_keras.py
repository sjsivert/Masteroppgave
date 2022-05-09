import logging
from abc import ABC
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from src.data_types.cnn_ae_lstm_model_keras import CNNAELSTMModel
from src.data_types.lstm_keras_model import LstmKerasModel
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from src.utils.keras_error_calculations import (
    config_metrics_to_keras_metrics,
    generate_error_metrics_dict,
    keras_mase_periodic,
)
from src.utils.lr_scheduler import scheduler
from tensorflow.keras.callbacks import LambdaCallback


class CnnAELstmKerasGlobalModel(CNNAELSTMModel, ABC):
    def __init__(
        self,
        log_sources: List[ILogTrainingSource],
        time_series_ids: List[int],
        params: Dict,
        optuna_trial: Optional[optuna.trial.Trial] = None,
    ):
        super(CNNAELSTMModel, self).__init__(
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

        is_tuning = xargs.pop("is_tuning") if "is_tuning" in xargs else False

        # Training AE
        self.train_auto_encoder(is_tuning)

        # Training LSTM
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

        self.train_lstm(self.hyper_parameters["lstm"]["epochs"], is_tuning, callbacks=all_callbacks)
        return self.metrics

    def test(self, predictive_period: int = 7, single_step: bool = False) -> Dict:
        logging.info("Testing")
        self._copy_trained_weights_to_model_with_different_batch_size()

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

            # Test AE
            self.test_auto_encoder(x_test)

            # Test LSTM
            metrics = self.test_lstm(x_train, y_train, x_test, y_test)

            global_metrics[testing_set_name] = metrics.copy()
        self.metrics = global_metrics
        return self.metrics
