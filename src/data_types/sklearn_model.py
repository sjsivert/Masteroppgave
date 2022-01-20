import pickle
from typing import Dict, List

from pandas import DataFrame

from src.data_types.i_model import IModel
from src.save_experiment_source.i_log_training_source import ILogTrainingSource


class SklearnModel(IModel):
    """
    Wrapper for sklearn models.
    """

    def __init__(self, model: object, log_sources: List[ILogTrainingSource]) -> None:
        super().__init__(log_sources)
        self.model = model

    def train(self, data_set: DataFrame, epochs: int = 10) -> Dict:
        # TODO: Implement
        pass

    def test(self, data_set: DataFrame, predictive_period: int = 5) -> Dict:
        # TODO: Implement
        pass

    def visualize(self, title: str = "default_title"):
        # TODO: Implement
        pass

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @staticmethod
    def load(path: str) -> IModel:
        with open(path, "rb") as f:
            return SklearnModel(pickle.load(f))
