import logging
import random
from typing import Dict, List, Optional, Tuple

from genpipes.compose import Pipeline
from matplotlib.figure import Figure
from pandas.core.frame import DataFrame
from src.data_types.arima_model import ArimaModel
from src.data_types.i_model import IModel
from src.model_strutures.i_model_structure import IModelStructure
from src.pipelines.local_univariate_arima_pipeline import \
    local_univariate_arima_pipeline
from src.save_experiment_source.i_log_training_source import ILogTrainingSource


class LocalUnivariateArimaStructure(IModelStructure):
    def __init__(self, log_sources: List[ILogTrainingSource], order: Tuple[int, int, int]) -> None:
        self.log_sources = log_sources
        self.data_pipeline = None
        self.order: Tuple[int, int, int] = order
        self.metrics: Dict = {}

    def init_models(self, load: bool = False):
        """
        Initialize models in the structure
        """
        self.models = [ArimaModel(order=self.order, log_sources=self.log_sources, name="1")]

    def process_data(self, data_pipeline: Pipeline) -> Optional[DataFrame]:
        self.data_pipeline = local_univariate_arima_pipeline(data_pipeline)
        logging.info(f"data preprocessing steps: \n {self.data_pipeline}")
        return self.data_pipeline.run()

    def get_data_pipeline(self) -> Pipeline:
        return self.data_pipeline

    def train(self) -> IModelStructure:
        # TODO: Do something with the training metrics returned by 'train' method
        # TODO: Pass training data to the 'train' method
        placeholder_data = DataFrame([random.randint(1, 25) for x in range(50)])
        for model in self.models:
            model.train(placeholder_data)

    def test(self) -> Dict:
        # TODO: Do something with the test metrics data returned by the 'test' method
        # TODO: Pass test data to the 'test' method
        placeholder_test_data = DataFrame([random.randint(1, 25) for x in range(5)])
        for model in self.models:
            model.test(placeholder_test_data)

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
