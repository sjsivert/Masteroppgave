import logging
from typing import Dict, List, Optional, Tuple

from genpipes.compose import Pipeline
from pandas.core.frame import DataFrame
from src.model_strutures.i_model_structure import IModelStructure
from src.pipelines.local_univariate_arima_pipeline import local_univariate_arima_pipeline
from src.save_experiment_source.i_log_training_source import ILogTrainingSource


class LocalUnivariateArimaStructure(IModelStructure):
    def __init__(self, log_sources: List[ILogTrainingSource], order: Tuple[int]) -> None:
        # TODO: Implement
        self.data_pipeline = None
        self.order = order

    def init_models(self, load: bool = False):
        """
        Initialize models in the structure
        """
        pass

    def process_data(self, data_pipeline: Pipeline) -> Optional[DataFrame]:
        self.data_pipeline = local_univariate_arima_pipeline(data_pipeline)
        logging.info(f"data preprocessing steps: \n {self.data_pipeline}")
        return self.data_pipeline.run()

    def get_data_pipeline(self) -> Pipeline:
        return self.data_pipeline

    def train(self) -> IModelStructure:
        # TODO: Implement
        pass

    def test(self) -> Dict:
        # TODO: Implement
        pass

    def get_metrics(self) -> Dict:
        # TODO: Implement
        raise NotImplementedError()
