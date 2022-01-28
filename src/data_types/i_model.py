from __future__ import annotations
from typing import List, Dict
from pandas.core.series import Series

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pandas.core.frame import DataFrame
from src.save_experiment_source.i_log_training_source import ILogTrainingSource


class IModel:
    """
    Interface for all models to implement in order to save and load
    """

    def __init__(self, log_sources: List[ILogTrainingSource], name: str = "placeholder"):
        # Temporary logging
        self.log_sources: List[ILogTrainingSource] = log_sources
        # Model name
        self.name: str = name
        # Predictions
        self.training_value_aproximation: Series = Series()
        self.predictions: Series = Series()
        # Visualization
        self.figures: List[Figure] = []
        self.metrics: Dict = {}

    def get_name(self):
        return self.name

    def train(self, data_set: DataFrame, epochs: int = 10) -> Dict:
        """
        Train the model.
        Return dict with training accuracy and error metric
        """
        raise NotImplementedError()

    def test(self, data_set: DataFrame, predictive_period: int = 5) -> Dict:
        """
        Test the trained model with test set
        Validate through prediction
        """
        raise NotImplementedError()

    def get_figures(self) -> List[Figure]:
        """
        Return a list of figures created by the model for visualization
        """
        pass

    def get_metrics(self) -> Dict:
        """
        Fetch metrics from model training or testing
        """
        raise NotImplementedError()

    def save(self, path: str) -> str:
        """
        Save the model to the specified path.
        :returns: Path to saved model file
        """
        raise NotImplementedError()

    def load(self, path: str) -> IModel:
        """
        Load the model from the specified path.
        """
        raise NotImplementedError()
