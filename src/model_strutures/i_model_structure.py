from __future__ import annotations

from typing import Dict, Optional, List

from genpipes.compose import Pipeline
from pandas import DataFrame

from src.data_types.i_model import IModel
from src.save_experiment_source.i_log_training_source import ILogTrainingSource


class IModelStructure:
    """
    Interface for all model types.
    Contains methods shared by all.
    """

    def __init__(self, log_sources: List[ILogTrainingSource], model_options: Dict):
        # This is an interface, so it should not be instantiated.
        pass

    def process_data(self, data_pipeline: Pipeline) -> Optional[DataFrame]:
        """
        Processes data to get it on the correct format for the relevant model.
        args:
          data_pipeline: Pipeline object containing the data to be processed.
        """
        pass

    def train(self) -> IModelStructure:
        """
        Trains the model.
        """
        pass

    def test(self) -> Dict:
        """
        Tests the model.
        """
        pass

    def get_models(self) -> List[IModel]:
        """
        Return the modes contained in the structure
        """
        pass
