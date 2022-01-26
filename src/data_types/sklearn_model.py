import pickle
from typing import Dict, List, Optional

from pandas import DataFrame
from src.data_types.i_model import IModel
from src.save_experiment_source.i_log_training_source import ILogTrainingSource


class SklearnModel(IModel):
    """
    Wrapper for sklearn models.
    """

    def __init__(
        self,
        model: Optional[object] = None,
        log_sources: List[ILogTrainingSource] = [],
        name: str = "sklearn_placeholder",
    ) -> None:
        self.model = model
        super().__init__(log_sources, name)

    def save(self, path: str) -> str:
        """
        Saves model
        :return:
        """
        save_path = f"{path}model_{self.get_name()}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(self.model, f)
        return save_path

    def load(self, path: str) -> None:
        with open(f"{path}model_{self.get_name()}.pkl", "rb") as f:
            self.model = pickle.load(f)

    def train(self, data_set: DataFrame, epochs: int = 10) -> Dict:
        # TODO: Implement
        raise NotImplementedError()

    def test(self, data_set: DataFrame, predictive_period: int = 5) -> Dict:
        # TODO: Implement
        raise NotImplementedError()

    def get_metrics(self) -> Dict:
        # TODO: Implement
        raise NotImplementedError()
