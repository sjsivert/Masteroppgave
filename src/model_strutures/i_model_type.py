from __future__ import annotations

from typing import Dict

from genpipes.compose import Pipeline
from pandas import DataFrame


class IModelType:
    """
    Interface for all model types.
    Contains methods shared by all.
    """

    def __init__(self, model_options: Dict):
        # This is an interface, so it should not be instantiated.
        pass

    def process_data(self, data_pipeline: Pipeline) -> DataFrame:
        """
        Processes data to get it on the correct format for the relevant model.
        args:
          data_pipeline: Pipeline object containing the data to be processed.
        """
        pass

    def train_model(self) -> IModelType:
        """
        Trains the model.
        """
        pass

    def test_model(self) -> Dict:
        """
        Tests the model.
        """
        pass
