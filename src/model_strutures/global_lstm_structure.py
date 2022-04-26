import logging
import typing
from typing import Any, List, Optional, OrderedDict, Tuple

from src.data_types.Lstm_keras_global_model import LstmKerasGlobalModel
from src.data_types.lstm_model import LstmModel
from src.model_strutures.neural_net_model_structure import NeuralNetworkModelStructure
from src.save_experiment_source.i_log_training_source import ILogTrainingSource


class GlobalLstmStructure(NeuralNetworkModelStructure):
    def __init__(
        self,
        log_sources: List[ILogTrainingSource],
        model_structure: OrderedDict[str, Any],
        datasets: List[str],
        hyperparameter_tuning_range: Optional[OrderedDict[str, Tuple[int, int]]] = None,
    ):
        super().__init__(log_sources, hyperparameter_tuning_range)
        self.datasets = datasets
        self.parameters_for_all_models = model_structure
        self.tuning_parameter_error_sets = None

    def init_models(self, load: bool = False):
        self.models.append(
            LstmKerasGlobalModel(
                log_sources=self.log_sources,
                params=self.parameters_for_all_models,
                time_series_ids=self.datasets,
            )
        )

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
                metric=None,
            )
            self.tuning_parameter_error_sets = {f"{base_model.get_name()}": best_trial}
