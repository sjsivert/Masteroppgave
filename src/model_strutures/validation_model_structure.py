from pathlib import Path
from typing import Dict, List, Optional

from genpipes.compose import Pipeline
from matplotlib.figure import Figure
from pandas.core.frame import DataFrame

from src.data_types.i_model import IModel
from src.data_types.validation_model import ValidationModel
from src.model_strutures.i_model_structure import IModelStructure
from src.save_experiment_source.i_log_training_source import ILogTrainingSource


class ValidationModelStructure(IModelStructure):
    def __init__(self, log_sources: List[ILogTrainingSource]) -> None:
        self.models = [ValidationModel(log_sources)]
        self.log_sources = log_sources
        self.figures = []
        self.training_metrics = {}
        self.testing_metrics = {}

    def process_data(self, data_pipeline: Pipeline) -> Optional[DataFrame]:
        # Get the data_set from the pipeline -> The pipeline runs as intended, returning a pipeline
        data_set = data_pipeline.run()
        return data_set

    def train(self) -> IModelStructure:
        print("Training is conducted, and training metrics are set.")
        print("Temporary metrics are logged, and model is saved.")
        # The model should complete training and save complete metrics, create mock metrics
        for idx, model in enumerate(self.models):
            self.training_metrics[f"model_{idx}"] = model.train(DataFrame(), epochs=5)
        return self

    def test(self) -> Dict:  # Return metrics
        # Test model in two steps
        # 1. Make prediction on training set in order to evaluate ability to fit the data
        # 2. Make prediction on the test set in order to evaluate ability to predict and generalize
        # 3. Save the prediction to the object and return metrics
        print("Testing is conducted, and prediction values and metrics are set and stored.")
        for idx, model in enumerate(self.models):
            self.testing_metrics[f"model_{idx}"] = model.test(DataFrame(), predictive_period=5)
        return self.testing_metrics

    def get_models(self) -> List[IModel]:
        return self.models

    def visualize(self) -> List[Figure]:  # Returns integer number of figures created
        self.figures = []
        for model in self.models:
            self.figures.extend(model.visualize())
        return self.figures

    def load_models(self, model_paths: List[Path]) -> None:
        self.models = map(lambda model_path: ValidationModel.load(model_path), model_paths)
