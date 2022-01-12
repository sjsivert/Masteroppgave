from typing import Dict, List
from matplotlib.axes import Axes


class ISaveExperimentSource:
    """
    An interface for all experiment save sources to implement.
    For example, disk, database, neptune.ai, etc.
    """

    def __init__(self) -> None:
        # Interface, not to be implemented
        pass

    def save_options(self, options: str) -> None:
        # Interface, not to be implemented
        pass

    def save_models(self, models: List) -> None:
        # Interface, not to be implemented
        pass

    def load_models(self, models_path: List) -> None:
        # Interface, not to be implemented
        return None

    def save_metrics(self, metrics: Dict[str, Dict[str, float]]) -> None:
        # Interface, not to be implemented
        pass

    def save_figures(self, figures: List[Axes]) -> None:
        # Saves pyplot axes
        pass
