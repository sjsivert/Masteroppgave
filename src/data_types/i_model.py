from __future__ import annotations

from typing import List, Dict
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

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
        self.training_accuracy: List = []
        self.training_error: List = []
        # Testing
        self.testing_error = None
        self.testing_accuracy = None
        self.testing_predictions = []
        # Visualization
        self.figures = []

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
        return self.figures

    def visualize(self, title: str = "default_title") -> None:
        """
        Visualize data attained from training and testing of the system
        """
        self.figures = []
        self._visualize_training_accuracy(title=title)
        self._visualize_training_error(title=title)
        self._visualize_prediction(title=title)
        print("Figures:", len(self.figures))

    def _visualize_training_accuracy(self, title: str) -> None:
        # Visualize training accuracy figure
        fig_1 = plt.figure(num=f"{title}_accuracy")
        self.figures.append(fig_1)
        plt.clf()
        plt.title(f"{title} - Training accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuarcy")
        plt.plot(self.training_accuracy, label="Accuracy")
        plt.close()

    def _visualize_training_error(self, title: str) -> None:
        # Visualize training error figure
        fig_2 = plt.figure(num=f"{title}_error")
        self.figures.append(fig_2)
        plt.clf()
        plt.title(f"{title} - Training error")
        plt.xlabel("Epochs")
        plt.ylabel("Error")
        plt.plot(self.training_error, label="Error")
        plt.close()

    def _visualize_prediction(self, title: str) -> None:
        # TODO: Visualize predictions
        # TODO: Visualize adaptability to original training data
        # TODO: Implement after it is clear how data is processed and what data is passed through to the model
        pass

    def save(self, path: str) -> None:
        """
        Save the model to the specified path.
        """
        raise NotImplementedError()

    @staticmethod
    def load(path: str, log_sources: List[ILogTrainingSource]) -> IModel:
        """
        Load the model from the specified path.
        """
        raise NotImplementedError()
