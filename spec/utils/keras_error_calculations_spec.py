import confuse
from confuse import Configuration
from confuse.exceptions import NotFoundError
from expects import be_true, equal, expect
from mamba import after, before, description, included_context, it, shared_context
from mockito import unstub, verify, when
from pandas import DataFrame
from spec.mock_config import init_mock_config
from src.utils import config_parser
from src.utils.config_parser import config
from src.utils.error_calculations import *
from src.utils.keras_error_calculations import (
    config_metrics_to_keras_metrics,
    generate_error_metrics_dict,
)

with description("Keras_error_calculations.py", "this") as self:

    with it("generate_error_metrics_dict works as expected"):
        init_mock_config()
        # Arrange
        errors = [0.1, 0.2, 0.3]
        metric_names = ["mse", "mae", "mape"]
        config["experiment"]["error_metrics"] = metric_names

        expected_result = {"test_mse": 0.1, "test_mae": 0.2, "test_mape": 0.3}

        # Act
        result = generate_error_metrics_dict(errors)

        # Assert
        expect(result).to(equal(expected_result))

    with it("config_metrics_to_keras_metrics works as expected"):
        init_mock_config()
        # Arrange
        metric_names = ["MSE", "MAE", "MASE", "SMAPE"]
        config["experiment"]["error_metrics"] = metric_names

        expected_result = [
            "mean_squared_error",
            "mean_absolute_error",
            "mean_absolute_scaled_error",
            "symetric_mean_absolute_percentage_error",
        ]

        # Act
        result = config_metrics_to_keras_metrics()

        expect(result).to(equal(expected_result))
