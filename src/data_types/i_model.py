from __future__ import annotations

from typing import List, Dict

from pandas.core.frame import DataFrame

from src.save_experiment_source.i_log_training_source import ILogTrainingSource


class IModel:
    """
    Interface for all models to implement in order to save and load
    """

    def __init__(self, log_sources: List[ILogTrainingSource]):
        # Temporary logging
        self.log_sources = log_sources
        # Training
        self.training_accuracy = []
        self.training_error = []
        # Testing
        self.testing_error = None
        self.testing_accuracy = None
        self.testing_predictions = []

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

    def visualize(self):
        """
        Visualize data attained from training and testing of the system
        """
        raise NotImplementedError()

    def save(self, path: str) -> None:
        """
        Save the model to the specified path.
        """
        raise NotImplementedError()

    @staticmethod
    def load(path: str) -> IModel:
        """
        Load the model from the specified path.
        """
        raise NotImplementedError()
