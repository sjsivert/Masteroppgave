import logging
import typing
from typing import List, OrderedDict, Optional, Any, Tuple

from src.data_types.cnn_ae_model import CNNAEModel
from src.data_types.lstm_model import LstmModel
from src.model_strutures.neural_net_model_structure import NeuralNetworkModelStructure
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from genpipes.compose import Pipeline


class LocalUnivariateCNNAELSTMStructure(NeuralNetworkModelStructure):
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
        super().__init__(
            log_sources,
            model_structure,
            common_parameters_for_all_models,
            hyperparameter_tuning_range,
            metric_to_use_when_tuning,
        )

    def init_models(self, load: bool = False):
        hyperparameters = self.common_parameters_for_all_models.copy()

        for model_structure in self.model_structure:
            hyperparameters.update(model_structure)
            model = CNNAEModel(
                log_sources=self.log_sources,
                params=hyperparameters,
                time_series_id=model_structure["time_series_id"],
            )
            self.models.append(model)

    def auto_tuning(self) -> None:
        """
        Automatic tuning of the model
        """
        # TODO:
        pass
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
        """
