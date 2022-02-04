from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import List, Dict
from pandas.core.series import Series

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pandas.core.frame import DataFrame
from src.save_experiment_source.i_log_training_source import ILogTrainingSource


class IModel(metaclass=ABCMeta):
    """
    Interface for all models to implement in order to save and load
    """

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def train(self, data_set: DataFrame, epochs: int = 10) -> Dict:
        """
        Train the model.
        Return dict with training accuracy and error metric
        """
        pass

    @abstractmethod
    def test(self, data_set: DataFrame, predictive_period: int = 5) -> Dict:
        """
        Test the trained model with test set
        Validate through prediction
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
