import logging
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


class LocalUnivariateLstmStructure(IModelStructure, ABC):
    def __init__(
        self,
        log_sources: List[ILogTrainingSource],
        model_structure: List,
        common_parameters_for_all_models: OrderedDict[str, Any],
        hyperparameter_tuning_range: Optional[OrderedDict[str, Tuple[int, int]]] = None,
        # steps_to_predict: int = 5,
        # multi_step_forecast: bool = False,
        # metric_to_use_when_tuning: str = "SMAPE",
    ):
        super().__init__()
        self.log_sources = log_sources
        self.common_parameters_for_all_models = common_parameters_for_all_models
        self.data_pipeline: Pipeline
        self.model_structure = model_structure
        self.hyperparameter_tuning_range = hyperparameter_tuning_range

        self.models: List[IModel] = []

    def init_models(self, load: bool = False):
        self.models = list(
            map(
                lambda model_structure: LstmModel(
                    log_sources=self.log_sources,
                    **model_structure,
                    **self.common_parameters_for_all_models,
                ),
                self.model_structure,
            )
        )

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
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(),
        )
        parameter_space = self.hyperparameter_tuning_range
        # TODO Make number of trials configuable
        study.optimize(
            lambda trial: local_univariate_lstm_objective(
                trial,
                parameter_space,
            ),
            n_trials=15,
        )
        pass

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
        # TODO: Implement
        return {}

    def get_predictions(self) -> Optional[DataFrame]:
        """
        Returns the predicted values if test() has been called.
        """
        # TODO: Implement
        return None
