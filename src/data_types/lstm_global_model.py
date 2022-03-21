import logging
from abc import ABC
from typing import List, Dict, Optional

import optuna
from pandas import DataFrame

from src.data_types.lstm_model import LstmModel
from src.pipelines.global_univariate_lstm_pipeline import global_univariate_lstm_pipeline
from src.save_experiment_source.i_log_training_source import ILogTrainingSource


class LstmGlobalModel(LstmModel, ABC):
    def __init__(
        self,
        log_sources: List[ILogTrainingSource],
        time_series_ids: List[str],
        params: Dict,
        optuna_trial: Optional[optuna.trial.Trial] = None,
    ):
        print("timeseries ids", time_series_ids)
        self.name = ",".join(map(str, time_series_ids))
        self.time_series_ids = time_series_ids

        super().__init__(
            log_sources=log_sources,
            time_series_id=self.name,
            params=params,
            optuna_trial=optuna_trial,
        )

    def process_data(self, data_set: DataFrame, training_size: float) -> None:
        data_pipeline = global_univariate_lstm_pipeline(
            data_set=data_set,
            cat_ids=self.time_series_ids,
            input_window_size=self.input_window_size,
            output_window_size=self.output_window_size,
        )

        logging.info(f"Data Pipeline for {self.get_name()}: {data_pipeline}")

        for log_source in self.log_sources:
            log_source.log_pipeline_steps(data_pipeline.__repr__())

        (
            self.training_dataset,
            self.validation_dataset,
            self.testing_dataset,
            self.min_max_scaler,
        ) = data_pipeline.run()

        self._convert_dataset_to_dataloader(
            self.training_dataset,
            self.validation_dataset,
            self.testing_dataset,
            batch_size=self.batch_size,
        )
