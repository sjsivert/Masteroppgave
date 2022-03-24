from enum import Enum


class ModelStructureEnum(Enum):
    validation_model = "validation_model_structure"
    local_univariate_arima = "local_univariate_arima_structure"
    univariate_lstm = "univariate_lstm_structure"
