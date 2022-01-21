from pathlib import Path
from typing import Dict, List

from matplotlib.figure import Figure

from src.data_types.i_model import IModel


class ISaveExperimentSource:
    """
    An interface for all experiment save sources to implement.
    For example, disk, database, neptune.ai, etc.
    """

    def save_model_and_metadata(
        self,
        options: str,
        metrics: Dict[str, Dict[str, float]],
        datasets: Dict[str, str],
        models: List[IModel],
        figures: List[Figure],
        data_pipeline_steps: str,
        experiment_tags: List[str],
    ) -> None:
        # Interface, not to be implemented
        pass

    def __init__(self) -> None:
        # Interface, not to be implemented
        pass

    def _save_options(self, options: str) -> None:
        # Interface, not to be implemented
        pass

    def _save_models(self, models: List[IModel]) -> None:
        # Interface, not to be implemented
        pass

    def _save_dataset_version(self, datasets: Dict[str, str]) -> None:
        # Interface
        pass

    @staticmethod
    def _load_models(models_path: List[Path]) -> None:
        # Interface, not to be implemented
        return None

    def _save_metrics(self, metrics: Dict[str, Dict[str, float]]) -> None:
        # Interface, not to be implemented
        pass

    def _save_figures(self, figures: List[Figure]) -> None:
        # Saves pyplot axes
        pass

    def _save_data_pipeline_steps(self, data_pipeline_steps: str) -> None:
        # Saves the steps of the pipeline
        pass
