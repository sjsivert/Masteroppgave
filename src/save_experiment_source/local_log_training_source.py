import csv
import os
from abc import ABC
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pipe
from pipe import map, tee, where

from src.data_types.i_model import IModel
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from src.save_experiment_source.save_local_disk_source import SaveLocalDiskSource


class LocalLogTrainingSource(SaveLocalDiskSource, ILogTrainingSource, ABC):
    def __init__(
        self,
        model_save_location: Path,
        title: str,
        description: str = "",
        options_dump: str = "",
        load_from_checkpoint: bool = False,
    ):
        super().__init__(
            model_save_location, title, description, options_dump, load_from_checkpoint
        )
        self.log_location = Path(f"{self.save_location}/logging/")
        self.epoch_counter = 0

    def log_metrics(self, metrics: Dict[str, Dict[str, float]], epoch: int) -> None:
        """
        Appends ../training_errors.csv in the current format
           epoch         model_id  MAE   MSE
        0      0              GPU   42  0.69
        1      0  Nettverkskabler  420  1.00
        """
        with self._create_log_folder_if_not_exist():
            with self._create_training_errors_file_if_not_exist(metrics):
                with open(f"{self.log_location}/training_errors.csv", "a") as f:
                    for cat_id, metric_values in metrics.items():
                        f.write(f"{epoch},{cat_id},")
                        metric_values_str = list(
                            metric_values.values() | pipe.map(lambda x: str(x))
                        )
                        f.write(",".join(list(metric_values_str)))
                        f.write("\n")

    def log_models(self, models: List[IModel]) -> None:
        with self._create_log_folder_if_not_exist():
            self._save_models(models, custom_save_path=self.log_location)

    def load_temp_models(self, models_path: List) -> None:
        # TODO: Implement
        raise NotImplementedError()

    def log_tuning_metrics(self, metrics: Dict[str, Dict[str, float]]):
        with self._create_log_folder_if_not_exist():
            with self._create_tuning_metric_file_if_not_exist():
                with open(f"{self.log_location}/tuning_metrics.csv", "a") as f:
                    for data_set, val in metrics.items():
                        for param, err in val.items():
                            f.write(f"{data_set},{param},{err}\n")

    def load_tuning_metrics(self) -> Dict[str, Dict[str, float]]:
        if os.path.isfile(f"{self.log_location}/tuning_metrics.csv"):
            with open(f"{self.log_location}/tuning_metrics.csv") as f:
                tuning_metrics = {}
                reader = csv.DictReader(f)
                for row in reader:
                    if row["dataset"] not in tuning_metrics:
                        tuning_metrics[row["dataset"]] = {}
                    tuning_metrics[row["dataset"]][row["parameters"]] = row["errorvalue"]
                return tuning_metrics
        return None

    @contextmanager
    def _create_log_folder_if_not_exist(self):
        try:
            os.mkdir(f"{self.log_location}")
        except FileExistsError:
            pass
        finally:
            yield

    @contextmanager
    def _create_tuning_metric_file_if_not_exist(self):
        if not os.path.isfile(f"{self.log_location}/tuning_metrics.csv"):
            with open(f"{self.log_location}/tuning_metrics.csv", "w") as f:
                f.write("dataset,parameters,errorvalue")
                f.write("\n")
        yield

    @contextmanager
    def _create_training_errors_file_if_not_exist(self, metrics: Dict[str, Dict[str, float]]):
        if not os.path.isfile(f"{self.log_location}/training_errors.csv"):
            metrics = LocalLogTrainingSource.extract_all_error_metrics_from_dict(metrics)
            with open(f"{self.log_location}/training_errors.csv", "w") as f:
                f.write("epoch,model_id,")
                f.write(",".join(metrics))
                f.write("\n")
        yield

    @staticmethod
    def extract_all_error_metrics_from_dict(metrics: Dict[str, Dict[str, float]]) -> List[str]:
        error_metrics = list(
            list(metrics.keys())
            | pipe.map(lambda cat_id: list(metrics[cat_id].keys()))
            | pipe.chain  # flatten list
            | pipe.dedup  # take only unique values
        )
        return error_metrics
