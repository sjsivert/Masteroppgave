import confuse
import tensorflow as tf
from confuse import Configuration
from confuse.exceptions import NotFoundError
from expects import be_true, equal, expect
from mamba import _it, after, before, description, included_context, it, shared_context
from mockito import unstub, verify, when
from pandas import DataFrame
from spec.mock_config import init_mock_config
from src.utils import config_parser
from src.utils.config_parser import config
from src.utils.error_calculations import *
from src.utils.keras_error_calculations import (
    config_metrics_to_keras_metrics,
    generate_error_metrics_dict,
    keras_mase,
    keras_smape,
)

with description("Keras_error_calculations.py", "unit") as self:

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
            keras_mase,
            keras_smape,
        ]

        # Act
        result = config_metrics_to_keras_metrics()

        expect(result).to(equal(expected_result))

    with _it("can caluclate MASE"):
        # TODO: Test if it calculates MASE correctly!
        data_set = np.array(
            [
                2,
                4,
                1,
            ]
        )
        proposed_data = np.array([3, 2, 1])
        expected_mase = 0.4
        mase = keras_mase(data_set, proposed_data)
        expect(round(mase, 4)).to(equal(expected_mase))
