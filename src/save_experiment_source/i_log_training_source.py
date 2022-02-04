from abc import ABCMeta
from typing import Dict, List


class ILogTrainingSource(metaclass=ABCMeta):
    """
    An interface for saving temporary or unfinished models and metrics.
    When training is interrupted, the model should be able to store the current training progress
    so that it can be continued at a later point.
    """

    def __init__(self) -> None:
        # Interface, not to be implemented
        pass

    def log_metrics(self, metrics: Dict[str, Dict[str, float]]) -> None:
        # Interface, not to be implemented
        pass

    def log_models(self, models: List) -> None:
        # Interface, not to be implemented
        pass

    def load_temp_models(self, models_path: List) -> None:
        # Interface, not to be implemented
        return None
