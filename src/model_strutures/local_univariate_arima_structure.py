import logging
import random
import warnings
from abc import ABC
from typing import Dict, List, Optional, OrderedDict, Tuple

import numpy as np
from genpipes.compose import Pipeline
from matplotlib.figure import Figure
from pandas.core.frame import DataFrame
from src.data_types.arima_model import ArimaModel
from src.data_types.error_metrics_enum import ErrorMetricEnum
from src.data_types.i_model import IModel
from src.model_strutures.i_model_structure import IModelStructure
from src.pipelines import local_univariate_arima_pipeline as arima_pipeline
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from src.utils.visuals import visualize_data_series


class LocalUnivariateArimaStructure(IModelStructure, ABC):
    def __init__(
        self,
        log_sources: List[ILogTrainingSource],
        forecast_window_size: float,
        model_structure: List,
        metric_to_use_when_tuning: str,
        hyperparameter_tuning_range: Optional[OrderedDict[str, Tuple[int, int]]] = None,
        steps_to_predict: int = 5,
        multi_step_forecast: bool = False,
        auto_arima: bool = True,
        seasonal: bool = True,
    ) -> None:
        self.models: List[IModel] = []
        self.log_sources = log_sources
        self.data_pipeline: Pipeline = None
        self.figures = []
        self.forecast_window_size = forecast_window_size
        self.model_structures = model_structure
        self.steps_to_predict = steps_to_predict
        self.multi_step_prediction = multi_step_forecast
        self.seasonal = seasonal

        self.metrics: Dict = {}
        # Data
        self.training_set: DataFrame = None
        self.testing_set: DataFrame = None
        # Tuning
        self.auto_arima = auto_arima
        self.tuning_parameter_error_sets = {}
        self.hyperparameter_tuning_range = hyperparameter_tuning_range
        self.metric_to_use_when_tuning = ErrorMetricEnum[metric_to_use_when_tuning]

    def init_models(self, load: bool = False):
        """
        Initialize models in the structure
        """
        self.models = list(
            map(
                lambda model_structure: ArimaModel(
                    log_sources=self.log_sources,
                    hyperparameters=model_structure["hyperparameters"],
                    name=model_structure["time_series_id"],
                    seasonal=self.seasonal,
                ),
                self.model_structures,
            )
        )

    def process_data(self, data_pipeline: Pipeline) -> Optional[DataFrame]:
        self.data_pipeline = data_pipeline

        logging.info(f"data preprocessing steps: \n {self.data_pipeline}")
        for log_source in self.log_sources:
            log_source.log_pipeline_steps(self.data_pipeline.__repr__())

        preprocessed_data = data_pipeline.run()

        for model in self.models:
            model.process_data(preprocessed_data, self.forecast_window_size)
        return self.training_set

    def train(self) -> IModelStructure:
        # TODO: Do something with the training metrics returned by 'train' method
        # TODO: Pass training data to the 'train' method
        for model in self.models:
            model.train()

    def test(self) -> Dict:
        # TODO: Do something with the test metrics data returned by the 'test' method
        # TODO: Pass test data to the 'test' method
        for model in self.models:
            model.test(
                predictive_period=self.steps_to_predict, single_step=not self.multi_step_prediction
            )

    # Exhaustive Grid Search of ARIMA model
    def auto_tuning(self) -> None:
        logging.info("Tuning models")
        self.figures = []

        all_parameters_to_test = self._generate_parameter_grid(
            p_range=self.hyperparameter_tuning_range["p"],
            d_range=self.hyperparameter_tuning_range["d"],
            q_range=self.hyperparameter_tuning_range["q"],
            P_range=self.hyperparameter_tuning_range["p"],
            D_range=self.hyperparameter_tuning_range["d"],
            Q_range=self.hyperparameter_tuning_range["q"],
            s_range=self.hyperparameter_tuning_range["s"],
        )
        logging.info(
            f"Parameter tuning ranges are: {self.hyperparameter_tuning_range} \n "
            f"Metric to use when tuning is: {self.metric_to_use_when_tuning}\n"
            f"Tuning parameter combinations to try are: {len(all_parameters_to_test)}# for each of the {len(self.model_structures)} datasets."
        )

        for base_model in self.models:
            logging.info(f"Tuning model: {base_model.get_name()}")
            # Auto ARIMA
            if self.auto_arima:
                error_parameter_set = base_model.auto_arima(
                    p=self.hyperparameter_tuning_range["p"][0],
                    d=self.hyperparameter_tuning_range["d"][0],
                    q=self.hyperparameter_tuning_range["q"][0],
                    max_p=self.hyperparameter_tuning_range["p"][1],
                    max_d=self.hyperparameter_tuning_range["d"][1],
                    max_q=self.hyperparameter_tuning_range["q"][1],
                    P=self.hyperparameter_tuning_range["P"][0],
                    D=self.hyperparameter_tuning_range["D"][0],
                    Q=self.hyperparameter_tuning_range["Q"][0],
                    max_P=self.hyperparameter_tuning_range["P"][1],
                    max_D=self.hyperparameter_tuning_range["D"][1],
                    max_Q=self.hyperparameter_tuning_range["Q"][1],
                    m=self.hyperparameter_tuning_range["s"][0],
                    epochs=100,
                    metric="mae",
                )
                self.tuning_parameter_error_sets[base_model.get_name()] = error_parameter_set
            # Grid search
            else:
                if str(base_model.get_name()) in self.tuning_parameter_error_sets:
                    logging.info(
                        f"{base_model.get_name()} was already tuned. Results can be found in the logg."
                    )
                    # Remove already tested parameters from the list
                    parameters = [
                        param
                        for param in all_parameters_to_test
                        if str(param)
                        not in self.tuning_parameter_error_sets[str(base_model.get_name())]
                    ]
                else:
                    parameters = all_parameters_to_test

                # Calculating Error
                error_parameter_sets = base_model.method_evaluation(
                    parameters,
                    metric=self.metric_to_use_when_tuning.value,
                    single_step=(not self.multi_step_prediction),
                )
                # Merge error sets
                if self.tuning_parameter_error_sets.get(base_model.get_name(), False):
                    self.tuning_parameter_error_sets[f"{base_model.get_name()}"].update(
                        error_parameter_sets
                    )
                else:
                    self.tuning_parameter_error_sets[base_model.get_name()] = error_parameter_sets
                for log_source in self.log_sources:
                    log_source.log_tuning_metrics(
                        {f"{base_model.get_name()}": error_parameter_sets}
                    )

    def _generate_parameter_grid(
        self,
        p_range: Tuple[int, int],
        d_range: Tuple[int, int],
        q_range: Tuple[int, int],
        P_range: Tuple[int, int],
        D_range: Tuple[int, int],
        Q_range: Tuple[int, int],
        s_range: Tuple[int, int],
    ) -> List[Tuple[int, int, int]]:
        parameters = []
        for q in range(q_range[0], q_range[1] + 1):
            for d in range(d_range[0], d_range[1] + 1):
                for p in range(p_range[0], p_range[1] + 1):
                    if self.seasonal:  # SARIMA parameters
                        for P in range(P_range[0], P_range[1] + 1):
                            for D in range(D_range[0], D_range[1] + 1):
                                for Q in range(Q_range[0], Q_range[1] + 1):
                                    for s in range(s_range[0], s_range[1] + 1):
                                        parameters.append(((p, d, q), (P, D, Q, s)))
                    else:  # ARIMA parameters
                        parameters.append(((p, d, q), None))
        return parameters

    def get_models(self) -> List[IModel]:
        return self.models

    def get_metrics(self) -> Dict:
        self.metrics = {}
        for model in self.models:
            self.metrics[model.get_name()] = model.get_metrics()
        return self.metrics

    def get_figures(self) -> List[Figure]:
        figures = []
        figures.extend(self.figures)
        for model in self.models:
            figures.extend(model.get_figures())
        return figures

    def get_tuning_parameters(self) -> Dict:
        return self.tuning_parameter_error_sets

    def get_predictions(self) -> DataFrame:
        predictions = {}
        for model in self.models:
            pred = model.get_predictions()
            if pred is not None:
                predictions[model.get_name()] = model.get_predictions()
        return DataFrame(predictions)

    def __repr__(self):
        return f"<Local_univariate_arim> models: {self.models}>"
