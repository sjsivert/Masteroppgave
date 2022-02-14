from abc import ABCMeta, abstractmethod
from typing import Dict, List


class ILogTrainingSource(metaclass=ABCMeta):
    """
    An interface for saving temporary or unfinished models and metrics.
    When training is interrupted, the model should be able to store the current training progress
    so that it can be continued at a later point.
    """

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Dict[str, float]], epoch: int) -> None:
        # Interface, not to be implemented
        pass

    @abstractmethod
    def log_models(self, models: List) -> None:
        # Interface, not to be implemented
        pass

    @abstractmethod
    def load_temp_models(self, models_path: List) -> None:
        # Interface, not to be implemented
        return None

    @abstractmethod
    def log_tuning_metrics(self, metrics: Dict[str, Dict[str, float]]):
        # Interface, not to be implemented
        pass

    @abstractmethod
    def load_tuning_metrics(self, metrics: Dict[str, Dict[str, float]]) -> None:
        # Interface, not to be implemented
        return None
