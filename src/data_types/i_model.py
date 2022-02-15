from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import List, Dict, Tuple, Optional

from matplotlib.figure import Figure
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from src.save_experiment_source.i_log_training_source import ILogTrainingSource


class IModel(metaclass=ABCMeta):
    """
    Interface for all models to implement in order to save and load
    """

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def process_data(
        self, data_set: DataFrame, training_size: float
    ) -> Tuple[DataFrame, DataFrame]:
        """
        Processes the already preprocessed data into training set and test set.
        :return:
        """

    @abstractmethod
    def train(self, epochs: int = 10) -> Dict:
        """
        Train the model.
        Return dict with training accuracy and error metric
        """
        pass

    @abstractmethod
    def test(self, predictive_period: int = 6, single_step: bool = False) -> Dict:
        """
        Test the trained model with test set
        Validate through prediction
        """
        pass

    @abstractmethod
    def method_evaluation(
        self,
        parameters: List,
        metric: str,
        singe_step: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate model and calculate error of predictions for used for tuning evaluation
        """
        pass

    @abstractmethod
    def get_figures(self) -> List[Figure]:
        """
        Return a list of figures created by the model for visualization
        """
        pass

    @abstractmethod
    def get_metrics(self) -> Dict:
        """
        Fetch metrics from model training or testing
        """
        pass

    @abstractmethod
    def save(self, path: str) -> str:
        """
        Save the model to the specified path.
        :returns: Path to saved model file
        """
        pass

    @abstractmethod
    def load(self, path: str) -> IModel:
        """
        Load the model from the specified path.
        """
        pass

    @abstractmethod
    def get_predictions(self) -> Optional[Dict]:
        """
        Returns the predicted values if test() has been called.
        """
        pass
