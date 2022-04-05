import logging
from abc import ABC
from typing import Any, Dict, List, Optional

import optuna
from src.data_types.neural_net_model import NeuralNetModel
from src.pipelines import local_multivariate_lstm_keras_pipeline as multivariate_pipeline
from src.pipelines import local_univariate_lstm_keras_pipeline as lstm_keras_pipeline
from src.save_experiment_source.i_log_training_source import ILogTrainingSource


class NeuralNetKerasModel(NeuralNetModel, ABC):
    def __init__(
        self,
        log_sources: List[ILogTrainingSource],
        time_series_id: str,
        params: Dict,
        optuna_trial: Optional[optuna.trial.Trial] = None,
    ):
        super(NeuralNetKerasModel, self).__init__(
            log_sources,
            time_series_id,
            params,
            optuna_trial,
            # pipeline=lstm_keras_pipeline.local_univariate_lstm_keras_pipeline,
            pipeline=multivariate_pipeline.local_multivariate_lstm_keras_pipeline,
            # pipeline= multivariate_pipeline.already_processed_dataset
        )
        self.should_shuffle_batches = params["should_shuffle_batches"]
        # Defining data set varaibles
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None

    def process_data(self, data_set: Any, training_size: float) -> None:
        data_pipeline = self.pipeline(
            data_set=data_set,
            cat_id=self.get_name(),
            training_size=self.training_size,
            input_window_size=self.input_window_size,
            output_window_size=self.output_window_size,
        )
        logging.info(f"Data Pipeline for {self.get_name()}: {data_pipeline}")
        for log_source in self.log_sources:
            log_source.log_pipeline_steps(data_pipeline.__repr__())

        self.training_data, self.testing_data, self.min_max_scaler = data_pipeline.run()
        self.split_data_sets()

    def split_data_sets(self):
        # Do not look a the code in the next 11 lines below, it is ugly and I am not proud of it
        examples_to_drop_to_make_all_batches_same_size = (
            self.training_data[0].shape[0] % self.batch_size
        )

        examples_to_drop_to_make_all_batches_same_size = (
            -self.hyper_parameters["output_window_size"]
            if examples_to_drop_to_make_all_batches_same_size == 0 and self.batch_size == 1
            else -examples_to_drop_to_make_all_batches_same_size
        )
        examples_to_drop_to_make_all_batches_same_size = (
            None
            if examples_to_drop_to_make_all_batches_same_size == 0
            else examples_to_drop_to_make_all_batches_same_size
        )

        logging.info(
            f"Examples to drop to make all batches same size: {examples_to_drop_to_make_all_batches_same_size}"
        )
        x_train, y_train = (
            self.training_data[0][:examples_to_drop_to_make_all_batches_same_size],
            self.training_data[1][:examples_to_drop_to_make_all_batches_same_size],
        )
        self.x_val, self.y_val = (
            x_train[-self.batch_size :],
            y_train[-self.batch_size :],
        )
        self.x_train = x_train[: -self.batch_size]
        self.y_train = y_train[: -self.batch_size]
        self.x_test, self.y_test = self.testing_data[0], self.testing_data[1]
