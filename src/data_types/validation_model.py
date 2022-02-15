from __future__ import annotations

from abc import ABC
from typing import List, Dict, Tuple, Optional

from pandas import Series
from pandas.core.frame import DataFrame
from matplotlib.figure import Figure

from src.data_types.i_model import IModel
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from src.utils.visuals import visualize_data_series


class ValidationModel(IModel, ABC):
    def __init__(self, log_sources: List[ILogTrainingSource], name: str = "validation_placeholder"):
        # Temporary logging
        self.log_sources: List[ILogTrainingSource] = log_sources
        # Model name
        self.name: str = name
        # Predictions
        self.training_value_aproximation: Series
        self.predictions: Series
        # Visualization
        self.figures: List[Figure] = []
        self.metrics: Dict = {}

    def get_name(self) -> str:
        return self.name

    def process_data(
        self, data_set: DataFrame, training_size: float
    ) -> Tuple[DataFrame, DataFrame]:
        # TODO: Implement
        pass

    def train(self, data_set: DataFrame, epochs: int = 10) -> Dict:
        """
        Set mock values for training accuracy and error
        """
        # Set mock values
        self.training_accuracy = [10, 20, 30, 37, 45]
        self.training_error = [14, 12, 11, 9, 8]

        self.figures.append(
            visualize_data_series(
                title="Training accuracy",
                data_series=[self.training_accuracy],
                data_labels=["Accuracy"],
                x_label="Date",
                y_label="Accuracy",
            )
        )
        self.figures.append(
            visualize_data_series(
                title="Training error",
                data_series=[self.training_error],
                data_labels=["Error"],
                x_label="Date",
                y_label="Error",
            )
        )
        self.metrics = {"Accuracy": self.training_accuracy[-1], "Error": self.training_error[-1]}
        return self.metrics

    def test(
        self, data_set: DataFrame, predictive_period: int = 5, single_step: bool = False
    ) -> Dict:
        """
        Mock values for testing accuracy, error, and predictions
        """
        self.testing_accuracy = 42
        self.testing_error = 9
        self.actual_values = [7, 9, 10, 12, 13]
        self.testing_predictions = [7, 8, 9, 10, 11]
        self.figures.append(
            visualize_data_series(
                title="Value predictions",
                data_series=[self.testing_predictions, self.actual_values],
                data_labels=["Prediction", "True values"],
                x_label="Date",
                y_label="Value",
            )
        )
        return {"Accuracy": self.testing_accuracy, "Error": self.testing_error}

    def method_evaluation(
        self,
        order: Tuple[int, int, int],
        metric: str,
        walk_forward: bool = True,
    ) -> float:
        # TODO: Implement
        raise NotImplementedError()

    def save(self, path: str) -> str:
        """
        Save the model to the specified path.
        :returns: Path to saved model file
        """
        save_path = f"{path}model_{self.get_name()}.pkl"
        with open(save_path, "w") as f:
            f.write("Validation model. Mock model saving.")
        return save_path

    def load(self, path: str) -> None:
        """
        Load the model from the specified path.
        """
        model_contents = ""
        with open(f"{path}model_{self.get_name()}.pkl", "r") as f:
            self.model_loaded_contents = f.read()

    def get_metrics(self) -> Dict:
        return self.metrics

    def get_figures(self) -> List[Figure]:
        return self.figures

    def get_predictions(self) -> Optional[Dict]:
        # TODO: Implement
        pass
