from __future__ import annotations

from typing import List, Dict

from pandas.core.frame import DataFrame

from src.data_types.i_model import IModel
from src.save_experiment_source.i_log_training_source import ILogTrainingSource


class ValidationModel(IModel):
    def __init__(self, log_sources: List[ILogTrainingSource], name: str = "validation_placeholder"):
        super().__init__(log_sources=log_sources, name=name)

    def train(self, data_set: DataFrame, epochs: int = 10) -> Dict:
        """
        Set mock values for training accuracy and error
        """
        # Set mock values
        self.training_accuracy = [10, 20, 30, 37, 45]
        self.training_error = [14, 12, 11, 9, 8]
        self.metrics = {"Accuracy": self.training_accuracy[-1], "Error": self.training_error[-1]}
        return self.metrics

    def test(self, data_set: DataFrame, predictive_period: int = 5) -> Dict:
        """
        Mock values for testing accuracy, error, and predictions
        """
        self.testing_accuracy = 42
        self.testing_error = 9
        self.testing_predictions = [7, 8, 9, 10, 11]
        return {"Accuracy": self.testing_accuracy, "Error": self.testing_error}

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
