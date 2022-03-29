from unittest.mock import ANY

import numpy as np
from expects import be, be_true, equal, expect, match
from genpipes.compose import Pipeline
from mamba import description, included_context, it, shared_context
from mockito import mock, when
from sklearn.preprocessing import MinMaxScaler
from spec.mock_config import init_mock_config
from spec.utils.mock_time_series_generator import mock_time_series_generator
from spec.utils.test_data import random_data_loader
from src.data_types.lstm_keras_model import LstmKerasModel
from src.data_types.modules.lstm_keras_module import LstmKerasModule
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from src.utils.config_parser import config
from src.utils.keras_optimizer import KerasOptimizer

with description(LstmKerasModule, "this") as self:
    with shared_context("init_lstm"):
        batch_size = 3
        model = LstmKerasModule(
            optimizer_name="Adam",
            stateful=False,
            output_window_size=1,
            number_of_layers=3,
            batch_size=batch_size,
            input_window_size=1,
            number_of_features=1,
            hidden_layer_size=64,
            recurrent_dropout=0.0,
            learning_rate=0.001,
            dropout=0.0,
            stateful_lstm=False,
        ).model

    with it("should be instantiable"):
        with included_context("init_lstm"):
            pass

    with it("should be able to train"):
        with included_context("init_lstm"):

            input_window_size = 1
            output_window_size = 1
            test_pipeline = mock_time_series_generator(
                input_window_size=input_window_size, output_window_size=output_window_size
            )
            X, Y, _ = test_pipeline.run()

            optimizer_name = "Adam"
            learning_rate = 0.001
            optim = KerasOptimizer.get(optimizer_name, learning_rate=learning_rate)

            model.compile(optimizer=optim, loss="mse", metrics=["mse"])
            model.fit(X, Y, batch_size=batch_size, epochs=1)

    # TODO: ---Move everything below to a separate test---
    with shared_context("init_lstm_model"):
        mock_config = init_mock_config()
        log_sources = [mock(ILogTrainingSource)]
        time_series_id = 1337
        common_params = config["model"]["local_univariate_lstm"][
            "common_parameters_for_all_models"
        ].get()
        hyper_params = config["model"]["local_univariate_lstm"]["model_structure"][0].get()
        common_params.update(
            hyper_params,
        )

        model = LstmKerasModel(
            log_sources=log_sources,
            time_series_id=str(time_series_id),
            params=common_params,
        )
        min_max_scaler = mock(MinMaxScaler)
        when(min_max_scaler, strict=False).inverse_transform(ANY).thenReturn(np.array([[1]]))
        model.min_max_scaler = min_max_scaler

    with it("should initialize LstmKerasModuel"):
        with included_context("init_lstm_model"):
            pass

    with it("should be able to train"):
        with included_context("init_lstm_model"):
            input_window_size = 7
            output_window_size = 1
            test_pipeline = mock_time_series_generator(
                input_window_size=input_window_size, output_window_size=output_window_size
            )
            X, Y, _ = test_pipeline.run()
            model.training_data = X, Y
            model.validation_data = X, Y

            model.train(epochs=1)
