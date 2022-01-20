import pickle
from typing import Dict

from pandas import DataFrame

from src.data_types.i_model import IModel


class SklearnModel(IModel):
    """
    Wrapper for sklearn models.
    """

    def train(self, data_set: DataFrame, epochs: int = 10) -> Dict:
        pass

    def test(self, data_set: DataFrame, predictive_period: int = 5) -> Dict:
        pass

    def visualize(self):
        pass

    def __init__(self, model: object) -> None:
        self.model = model

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @staticmethod
    def load(path: str) -> IModel:
        with open(path, "rb") as f:
            return SklearnModel(pickle.load(f))
