import os
import random
import shutil
from collections import OrderedDict
from datetime import datetime
from typing import Tuple

import pandas as pd
from expects import be_none, be_true, equal, expect
from ipywidgets import Datetime
from mamba import after, before, description, it, included_context, shared_context
from mockito.mocking import mock
from mockito.mockito import unstub
from pandas.core.frame import DataFrame
from spec.mock_config import init_mock_config
from src.data_types.arima_model import ArimaModel
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults


def split_into_test_and_training_set(
    df: DataFrame, training_size: float
) -> Tuple[DataFrame, DataFrame]:
    training_set = int((df.shape[0] - 1) * training_size)
    training_df = df[:training_set]
    testing_set = df[training_set:]
    return training_df, testing_set


with description(ArimaModel, "unit") as self:
    with before.all:
        log_source = mock(ILogTrainingSource)
        log_source.log_metrics = mock()
        self.log_sources = [log_source]
        self.temp_location = "spec/temp/"
        init_mock_config()
        try:
            os.mkdir(self.temp_location)
        except FileExistsError:
            pass

    with after.each:
        unstub()

    with shared_context("mock dataset"):
        mock_dataset = DataFrame(
            dict(
                interest=[random.randint(0, 100) for _ in range(29)],
            )
        )
        mock_training_set = mock_dataset[:80]
        mock_test_set = mock_dataset[80:]
        hyperparameters = OrderedDict([("p", 1), ("d", 0), ("q", 1)])

    with it("Can create ArimaModel object"):
        with included_context("mock dataset"):
            model = ArimaModel(
                hyperparameters=hyperparameters, log_sources=self.log_sources, name="createArima"
            )

    with it("Can train"):
        # Arrange
        with included_context("mock dataset"):
            model = ArimaModel(
                hyperparameters=hyperparameters, log_sources=self.log_sources, name="trainingArima"
            )
            model.training_data, model.test_data = split_into_test_and_training_set(
                df=mock_training_set, training_size=0.8
            )

            # Act
            train_metrics = model.train()
            # Assert
            expect(model.model).not_to(be_none)
            expect(len(model.value_approximation)).to(equal(len(model.training_data)))
            expect(("MAE" and "MSE") in train_metrics).to(be_true)

    with it("Can test"):
        # Arrange
        with included_context("mock dataset"):
            model = ArimaModel(
                hyperparameters=hyperparameters, log_sources=self.log_sources, name="testingArima"
            )
            # Act
            model.training_data, model.test_data = split_into_test_and_training_set(
                df=mock_training_set, training_size=0.8
            )
            model.train()
            test_metrics = model.test()
            # Assert
            expect(model.model).not_to(be_none)
            expect(len(model.predictions)).to(equal(5))
            expect(("MAE" and "MSE") in test_metrics).to(be_true)

    with it("can save"):
        # Arrange
        with included_context("mock dataset"):
            model = ArimaModel(
                hyperparameters=hyperparameters, log_sources=self.log_sources, name="savingArima"
            )
            model.training_data, model.test_data = split_into_test_and_training_set(
                df=mock_training_set, training_size=0.8
            )
            model.train()
            # Act
            path = f"{self.temp_location}Arima_{model.get_name()}.pkl"
            model.save(self.temp_location)
            # Assert
            expect(os.path.isfile(path)).to(be_true)

    with it("can load"):
        # Arrange
        with included_context("mock dataset"):
            model = ArimaModel(
                hyperparameters=hyperparameters, log_sources=self.log_sources, name="loadingArima"
            )
            model.training_data, model.test_data = split_into_test_and_training_set(
                df=mock_training_set, training_size=0.8
            )
            model.train()
            # Act
            path = f"{self.temp_location}Arima_{model.get_name()}.pkl"
            model.save(self.temp_location)

            loaded_model = ArimaModel(log_sources=self.log_sources, name="loadingArima")
            loaded_model.load(self.temp_location)
            loaded_model.training_data, loaded_model.test_data = split_into_test_and_training_set(
                df=mock_training_set, training_size=0.8
            )
            loaded_test_metrics = loaded_model.test()
            # Assert
            expect(loaded_model.model).to_not(be_none)
            expect(len(loaded_model.predictions)).to(equal(5))
            expect(("MAE" and "MSE") in loaded_test_metrics).to(be_true)
