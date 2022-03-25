from enum import Enum


class ModelStructureEnum(Enum):
    validation_model = "validation_model_structure"
    local_univariate_arima = "local_univariate_arima_structure"
    local_univariate_lstm = "local_univariate_lstm_structure"
    local_cnn_ae_lstm = "local_cnn_ae_lstm"
    local_cnn_ae = "local_cnn_ae"
