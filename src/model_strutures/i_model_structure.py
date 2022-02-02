from __future__ import annotations

from typing import Dict, List, Optional

from genpipes.compose import Pipeline
from matplotlib.figure import Figure
from pandas import DataFrame
from src.data_types.i_model import IModel
from src.save_experiment_source.i_log_training_source import ILogTrainingSource


class IModelStructure:
    """
    Interface for all model types.
    Contains methods shared by all.
    """

    def __init__(self, log_sources: List[ILogTrainingSource]):
        # This is an interface.
        pass

    def init_models(self, load: bool = False):
        """
        Initialize models in the structure
        """
        pass

    def process_data(self, data_pipeline: Pipeline) -> Optional[DataFrame]:
        """
        Processes data to get it on the correct format for the relevant model.
        args:
          data_pipeline: Pipeline object containing the data to be processed.
        """
        pass

    def get_data_pipeline(self) -> Pipeline:
        """
        Returns the data pipeline used to process the data.
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

    def auto_tuning(self, Dict) -> None:
        """
        Automatic tuning of the model
        """
        pass

    def get_models(self) -> List[IModel]:
        """
        Return the modes contained in the structure
        """
        pass

    def get_metrics(self) -> Dict:
        """
        Returns dict of metrics
        """
        pass

    def get_figures(self) -> List[Figure]:
        """
        Returns list of figures
        """
        pass

    def get_tuning_parameters(self) -> Dict:
        """
        Returns a dict with info regarding the automatic tuning of the models
        """
        pass
