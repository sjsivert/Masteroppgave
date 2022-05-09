import logging
import typing
from typing import Any, List, Optional, OrderedDict, Tuple, Dict

from src.data_types.cnn_ae_lstm_global_model_keras import CnnAELstmKerasGlobalModel
from src.data_types.lstm_keras_model import LstmKerasModel
from src.model_strutures.neural_net_model_structure import NeuralNetworkModelStructure
from src.save_experiment_source.i_log_training_source import ILogTrainingSource


class GlobalCnnAELstmStructure(NeuralNetworkModelStructure):
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
            CnnAELstmKerasGlobalModel(
                log_sources=self.log_sources,
                params=self.parameters_for_all_models,
                time_series_ids=self.datasets,
            )
        )

    def get_metrics(self) -> Dict:
        return self.models[0].get_metrics()

    def auto_tuning(self) -> None:
        """
        Automatic tuning of the model
        """
        for base_model in self.models:
            # Specify the class to compiler
            base_model = typing.cast(LstmKerasModel, base_model)

            logging.info(f"Tuning model: {base_model.get_name()}")

            best_trial = base_model.method_evaluation(
                parameters=self.hyperparameter_tuning_range,
                metric=None,
            )
            self.tuning_parameter_error_sets = {f"{base_model.get_name()}": best_trial}
