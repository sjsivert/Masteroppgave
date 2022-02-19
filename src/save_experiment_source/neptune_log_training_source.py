import os
import random
from abc import ABC
from typing import Dict, List

from neptune.new.attributes import File

from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from src.save_experiment_source.neptune_save_source import NeptuneSaveSource
from src.utils.temporary_files import temp_files


class NeptuneLogTrainingSource(NeptuneSaveSource, ILogTrainingSource, ABC):
    def __init__(
        self,
        project_id: str,
        title,
        description,
        load_from_checkpoint: bool = False,
        neptune_id_to_load: str = None,
        sync: bool = False,
        **xargs,
    ):
        super().__init__(
            project_id, title, description, load_from_checkpoint, neptune_id_to_load, sync, **xargs
        )

    def log_metrics(
        self, metrics: Dict[str, Dict[str, float]], epoch: int
    ) -> None:  # pragma: no cover
        for cat_id, metric_values in metrics.items():
            for metric_name, metric_value in metric_values.items():
                self.run[f"logging/{cat_id}/{metric_name}"].log(metric_value)

    def log_models(self, models: List) -> None:
        base_path = "./temp/"
        with temp_files(base_path):
            for model in models:
                model_save_path = model.save(f"{base_path}/model_{model.get_name()}")
                if os.path.isfile(model_save_path):
                    self.run[f"models/model_{model.get_name()}"].upload(File(model_save_path), True)

    def log_tuning_metrics(self, metrics: Dict[str, Dict[str, Dict[str, float]]]) -> None:
        for cat_id, metric_values in metrics.items():
            for param, metric_value in metric_values.items():
                self.run[f"logging_tuning/{cat_id}/{param}"].log(metric_value)

    def load_temp_models(self, models_path: List) -> None:
        return None

    def load_tuning_metrics(self) -> Dict[str, Dict[str, float]]:
        return self.run["logging_tuning"].fetch()
