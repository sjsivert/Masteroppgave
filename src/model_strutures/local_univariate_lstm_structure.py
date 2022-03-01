from abc import ABC
from typing import List, Optional, Dict, OrderedDict, Tuple

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
        training_size: float,
        model_structure: List,
        input_window_size: int,
        output_window_size: int,
        hyperparameter_tuning_range: Optional[OrderedDict[str, Tuple[int, int]]] = None,
        # steps_to_predict: int = 5,
        # multi_step_forecast: bool = False,
        # metric_to_use_when_tuning: str = "SMAPE",
    ):
        super().__init__()
        self.output_window_size = output_window_size
        self.input_window_size = input_window_size
        self.model_structure = model_structure
        self.training_size = training_size
        self.log_sources = log_sources
        self.hyperparameter_tuning_range = hyperparameter_tuning_range

    def init_models(self, load: bool = False):
        # TODO implement
        raise NotImplementedError()
        self.models = list(
            map(
                lambda model_structure: LstmModel(
                    log_sources=self.log_sources,
                    name=model_structure["time_series_id"],
                ),
                self.model_structure,
            )
        )

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
            lambda trial: local_univariate_lstm_objective(trial, parameter_space), n_trials=15
        )
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

    def get_predictions(self) -> Optional[DataFrame]:
        """
        Returns the predicted values if test() has been called.
        """
        pass
