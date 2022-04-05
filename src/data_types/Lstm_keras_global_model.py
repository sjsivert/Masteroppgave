from abc import ABC
from typing import List, Dict, Optional, Any

import numpy as np
import optuna

from src.data_types.lstm_keras_model import LstmKerasModel
from src.save_experiment_source.i_log_training_source import ILogTrainingSource


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

    def process_data(self, data_set: Any, training_size: float) -> None:
        cat_ids = [11573, 11037]
        x_train, y_train, x_val, y_val, x_test, y_test = [], [], [], [], [], []
        for cat_id in cat_ids:
            data_pipeline = self.pipeline(
                data_set=data_set,
                cat_id=cat_id,
                training_size=training_size,
                input_window_size=self.input_window_size,
                output_window_size=self.output_window_size,
            )
            for log_source in self.log_sources:
                log_source.log_pipeline_steps(data_pipeline.__repr__())
            training_data, testing_data, min_max_scaler = data_pipeline.run()
            training_data_splitted, validation_data, testing_data = self.split_data_sets(
                training_data, testing_data
            )
            x_train.append(training_data_splitted[0])
            y_train.append(training_data_splitted[1])
            x_val.append(validation_data[0])
            y_val.append(validation_data[1])
            x_test.append(testing_data[0])
            y_test.append(testing_data[1])

        self.x_train = np.concatenate(x_train, axis=0)
        self.y_train = np.concatenate(y_train, axis=0)
        self.x_val = np.concatenate(x_val, axis=0)
        self.y_val = np.concatenate(y_val, axis=0)
        self.x_test = np.concatenate(x_test, axis=0)
        self.y_test = np.concatenate(y_test, axis=0)
