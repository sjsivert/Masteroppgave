from abc import ABC
from pathlib import Path
from typing import Dict, List

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

    def log_metrics(self, metrics: Dict[str, Dict[str, float]]) -> None:
        pass

    def log_models(self, models: List) -> None:
        pass

    def load_temp_models(self, models_path: List) -> None:
        return None
