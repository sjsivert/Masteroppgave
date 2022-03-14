import logging
import typing
from abc import ABC
from typing import List, Optional, Dict, OrderedDict, Tuple, Any

import optuna
from genpipes.compose import Pipeline
from matplotlib.figure import Figure
from pandas import DataFrame

from src.data_types.i_model import IModel
from src.data_types.lstm_model import LstmModel
from src.model_strutures.i_model_structure import IModelStructure
from src.optuna_tuning.loca_univariate_lstm_objective import local_univariate_lstm_objective
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from src.utils.time_function import time_function


class LocalUnivariateLstmStructure(IModelStructure, ABC):
    def __init__(
        self,
        log_sources: List[ILogTrainingSource],
        model_structure: List,
        common_parameters_for_all_models: OrderedDict[str, Any],
        hyperparameter_tuning_range: Optional[OrderedDict[str, Tuple[int, int]]] = None,
        # steps_to_predict: int = 5,
        # multi_step_forecast: bool = False,
        metric_to_use_when_tuning: str = "MASE",
    ):
        super().__init__()
        self.tuning_parameter_error_sets = None
        self.metric_to_use_when_tuning = metric_to_use_when_tuning
        self.log_sources = log_sources
        self.common_parameters_for_all_models = common_parameters_for_all_models
        self.data_pipeline: Pipeline
        self.model_structure = model_structure
        self.hyperparameter_tuning_range = hyperparameter_tuning_range

        self.models: List[IModel] = []

    def init_models(self, load: bool = False):
        hyperparameters = self.common_parameters_for_all_models.copy()

        for model_structure in self.model_structure:
            hyperparameters.update(model_structure)
            model = LstmModel(
                log_sources=self.log_sources,
                params=hyperparameters,
                time_series_id=model_structure["time_series_id"],
            )
            self.models.append(model)

    def process_data(self, data_pipeline: Pipeline) -> None:
        """
        Processes data to get it on the correct format for the relevant model.
        args:
          data_pipeline: Pipeline object containing the data to be processed.
        """
        self.data_pipeline = data_pipeline

        logging.info(f"data preprocessing steps: \n {self.data_pipeline}")
        for log_source in self.log_sources:
            log_source.log_pipeline_steps(self.data_pipeline.__repr__())

        preprocessed_data = data_pipeline.run()

        with time_function():
            for model in self.models:
                model.process_data(preprocessed_data, 0)

    def train(self) -> IModelStructure:
        """
        Trains the model.
        """
        for model in self.models:
            model.train()

    def test(self) -> Dict:
        """
        Tests the model.
        """
        for model in self.models:
            model.test()

    def auto_tuning(self) -> None:
        """
        Automatic tuning of the model
        """
        for base_model in self.models:
            # Specify the class to compiler
            base_model = typing.cast(LstmModel, base_model)

            logging.info(f"Tuning model: {base_model.get_name()}")

            best_trial = base_model.method_evaluation(
                parameters=self.hyperparameter_tuning_range,
                metric=self.metric_to_use_when_tuning,
            )
            self.tuning_parameter_error_sets = {f"{base_model.get_name()}": best_trial}

    def get_models(self) -> List[IModel]:
        """
        Return the modes contained in the structure
        """
        return self.models

    def get_metrics(self) -> Dict:
        """
        Returns dict of metrics
        """
        metrics = {}
        for model in self.models:
            metrics[f"{model.get_name()}"] = model.get_metrics()
        return metrics

    def get_figures(self) -> List[Figure]:
        """
        Returns list of figures
        """
        figures = []
        for model in self.models:
            figures.extend(model.get_figures())
        return figures

    def get_tuning_parameters(self) -> Dict:
        """
        Returns a dict with info regarding the automatic tuning of the models
        """
        return self.tuning_parameter_error_sets

    def get_predictions(self) -> Optional[DataFrame]:
        """
        Returns the predicted values if test() has been called.
        """
        # TODO: Implement
        return None
