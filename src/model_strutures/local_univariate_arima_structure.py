from typing import Dict, Optional, Tuple, List

from genpipes.compose import Pipeline
from pandas.core.frame import DataFrame
from src.model_strutures.i_model_structure import IModelStructure
from src.save_experiment_source.i_log_training_source import ILogTrainingSource


class LocalUnivariateArimaStructure(IModelStructure):
    def __init__(self, log_sources: List[ILogTrainingSource], order: Tuple[int]) -> None:
        # TODO: Implement
        self.order = order

    def process_data(self, data_pipeline: Pipeline) -> Optional[DataFrame]:
        return data_pipeline.run()

    def train(self) -> IModelStructure:
        # TODO: Implement
        raise NotImplementedError()

    def test(self) -> Dict:
        # TODO: Implement
        raise NotImplementedError()
