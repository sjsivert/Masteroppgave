from abc import ABC
from typing import Dict, List

from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from src.save_experiment_source.neptune_save_source import NeptuneSaveSource


class NeptuneLogTrainingSource(NeptuneSaveSource, ILogTrainingSource, ABC):
    def __init__(
        self,
        project_id: str,
        title,
        description,
        load_from_checkpoint: bool = False,
        neptune_id_to_load: str = None,
        sync: bool = False,
        **xargs
    ):
        super().__init__(
            project_id, title, description, load_from_checkpoint, neptune_id_to_load, sync, **xargs
        )

    def log_metrics(self, metrics: Dict[str, Dict[str, float]]) -> None:
        pass

    def log_models(self, models: List) -> None:
        pass

    def load_temp_models(self, models_path: List) -> None:
        return None
