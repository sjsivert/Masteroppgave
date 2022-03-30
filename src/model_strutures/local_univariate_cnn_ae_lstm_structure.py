from typing import List, OrderedDict, Optional, Any, Tuple
from src.data_types.cnn_ae_lstm_model_keras import CNNAELSTMModel
from src.model_strutures.neural_net_model_structure import NeuralNetworkModelStructure
from src.save_experiment_source.i_log_training_source import ILogTrainingSource


class LocalUnivariateCNNAELSTMStructure(NeuralNetworkModelStructure):
    def __init__(
        self,
        log_sources: List[ILogTrainingSource],
        model_structure: List,
        common_parameters_for_all_models: OrderedDict[str, Any],
        hyperparameter_tuning_range: Optional[OrderedDict[str, Tuple[int, int]]] = None,
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
            model = CNNAELSTMModel(
                log_sources=self.log_sources,
                params=hyperparameters,
                time_series_id=model_structure["time_series_id"],
            )
            self.models.append(model)

    def auto_tuning(self) -> None:
        """
        Automatic tuning of the model
        """
        # TODO!
        pass
