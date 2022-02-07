import os
from abc import ABC
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Set

import pipe
from pipe import map, tee, where
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

    def log_metrics(self, metrics: Dict[str, Dict[str, float]]) -> None:
        """
        Appends ../training_errors.csv in the current format
           epoch         model_id  MAE   MSE
        0      0              GPU   42  0.69
        1      0  Nettverkskabler  420  1.00
        """
        with self._create_log_folder_if_not_exist():
            print(f"log folder exist: {os.path.isdir(f'{self.log_location}')}")
            with self._create_training_errors_file_if_not_exist(metrics):
                with open(f"{self.log_location}/training_errors.csv", "a") as f:
                    for cat_id, metric_values in metrics.items():
                        f.write(f"{self.epoch_counter},{cat_id},")
                        metric_values_str = list(
                            metric_values.values() | pipe.map(lambda x: str(x))
                        )
                        f.write(",".join(list(metric_values_str)))
                        f.write("\n")

                self.epoch_counter += 1

    def log_models(self, models: List) -> None:
        pass

    def load_temp_models(self, models_path: List) -> None:
        return None

    @contextmanager
    def _create_log_folder_if_not_exist(self):
        try:
            print(f"temp folder exist: {os.path.isdir(f'models/temp-log-training-source')}")
            print(
                f"model lcoation exist: {os.path.isdir(f'models/log-training-source/test-local-log-training-source')}"
            )
            os.mkdir(f"{self.log_location}")
        except FileExistsError:
            pass
        finally:
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
