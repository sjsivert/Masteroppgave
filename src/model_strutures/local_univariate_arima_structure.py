import logging
import random
from typing import Dict, List, Optional, Tuple

from genpipes.compose import Pipeline
from matplotlib.figure import Figure
from pandas.core.frame import DataFrame
from src.data_types.arima_model import ArimaModel
from src.data_types.i_model import IModel
from src.model_strutures.i_model_structure import IModelStructure
from src.pipelines.local_univariate_arima_pipeline import local_univariate_arima_pipeline
from src.save_experiment_source.i_log_training_source import ILogTrainingSource


class LocalUnivariateArimaStructure(IModelStructure):
    def __init__(
        self, log_sources: List[ILogTrainingSource], training_size: float, model_structure: List
    ) -> None:
        self.models = None
        self.log_sources = log_sources
        self.data_pipeline = None

        self.model_structures = model_structure
        self.training_size = training_size

        self.metrics: Dict = {}
        # Data
        self.training_set: DataFrame = None
        self.testing_set: DataFrame = None

    def init_models(self, load: bool = False):
        """
        Initialize models in the structure
        """
        self.models = list(
            map(
                lambda model_structure: ArimaModel(
                    log_sources=self.log_sources,
                    order=model_structure["order"],
                    name=model_structure["time_series_id"],
                    training_size=self.training_size,
                ),
                self.model_structures,
            )
        )

    def process_data(self, data_pipeline: Pipeline) -> Optional[DataFrame]:
        self.data_pipeline = local_univariate_arima_pipeline(data_pipeline)

        logging.info(f"data preprocessing steps: \n {self.data_pipeline}")
        self.training_set, self.testing_set = self.data_pipeline.run()
        return self.training_set

    def get_data_pipeline(self) -> Pipeline:
        return self.data_pipeline

    def train(self) -> IModelStructure:
        # TODO: Do something with the training metrics returned by 'train' method
        # TODO: Pass training data to the 'train' method
        for model in self.models:
            model.train(self.training_set)

    def test(self) -> Dict:
        # TODO: Do something with the test metrics data returned by the 'test' method
        # TODO: Pass test data to the 'test' method
        for model in self.models:
            model.test(self.testing_set, 10)

    def get_models(self) -> List[IModel]:
        return self.models

    def get_metrics(self) -> Dict:
        self.metrics = {}
        for model in self.models:
            self.metrics[model.get_name()] = model.get_metrics()
        return self.metrics

    def get_figures(self) -> List[Figure]:
        figures = []
        for model in self.models:
            figures.extend(model.get_figures())
        return figures
