import os
import shutil
import random

from expects import be_true, equal, expect, be_none
from mamba import after, before, description, it
from mockito.mocking import mock
from mockito.mockito import unstub
from pandas.core.frame import DataFrame
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

from src.data_types.arima_model import ArimaModel
from src.save_experiment_source.i_log_training_source import ILogTrainingSource

with description(ArimaModel, "unit") as self:

    with before.all:
        self.log_sources = [mock(ILogTrainingSource)]
        self.temp_location = "spec/temp/"
        try:
            os.mkdir(self.temp_location)
        except FileExistsError:
            pass

    with after.each:
        unstub()

    with it("Can create ArimaModel object"):
        order = (1, 1, 1)
        model = ArimaModel(order, self.log_sources, "createArima")

    with it("Can train"):
        # Arrange
        mock_dataset = DataFrame([random.randint(1, 40) for x in range(50)])
        order = (1, 1, 1)
        model = ArimaModel(order=order, log_sources=self.log_sources, name="trainingArima")
        # Act
        train_metrics = model.train(mock_dataset)
        # Assert
        expect(model.model).not_to(be_none)
        expect(len(model.value_approximation)).to(equal(len(mock_dataset)))
        expect(("MAE" and "MSE") in train_metrics).to(be_true)

    with it("Can test"):
        # Arrange
        mock_dataset = DataFrame([random.randint(1, 40) for x in range(50)])
        mock_test_dataset = DataFrame([random.randint(1, 40) for x in range(5)])
        order = (1, 1, 1)
        model = ArimaModel(order=order, log_sources=self.log_sources, name="testingArima")
        # Act
        model.train(mock_dataset)
        test_metrics = model.test(mock_test_dataset)
        # Assert
        expect(model.model).not_to(be_none)
        expect(len(model.predictions)).to(equal(5))
        expect(("MAE" and "MSE") in test_metrics).to(be_true)

    with it("can save"):
        # Arrange
        mock_dataset = DataFrame([random.randint(1, 40) for x in range(50)])
        order = (1, 1, 1)
        model = ArimaModel(order=order, log_sources=self.log_sources, name="savingArima")
        model.train(mock_dataset)
        # Act
        path = f"{self.temp_location}Arima_{model.get_name()}.pkl"
        model.save(self.temp_location)
        # Assert
        expect(os.path.isfile(path)).to(be_true)

    with it("can load"):
        # Arrange
        mock_dataset = DataFrame([random.randint(1, 40) for x in range(50)])
        mock_test_dataset = DataFrame([random.randint(1, 40) for x in range(5)])
        order = (1, 1, 1)
        model = ArimaModel(order=order, log_sources=self.log_sources, name="loadingArima")
        model.train(mock_dataset)
        # Act
        path = f"{self.temp_location}Arima_{model.get_name()}.pkl"
        model.save(self.temp_location)

        loaded_model = ArimaModel(log_sources=self.log_sources, name="loadingArima")
        loaded_model.load(self.temp_location)
        loaded_test_metrics = loaded_model.test(mock_test_dataset)
        # Assert
        expect(loaded_model.model).to_not(be_none)
        expect(len(loaded_model.predictions)).to(equal(5))
        expect(("MAE" and "MSE") in loaded_test_metrics).to(be_true)
