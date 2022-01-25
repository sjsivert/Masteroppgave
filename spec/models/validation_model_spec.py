import os
import shutil
from pathlib import Path

from expects import be_true, equal, expect, be_none
from genpipes.compose import Pipeline
from mamba import before, description, it

from mamba import after, before, description, it, shared_context, included_context
from mockito.mocking import mock
from mockito.mockito import when, verify, unstub
from mockito.matchers import ANY
from pandas.core.frame import DataFrame

from src.data_types.validation_model import ValidationModel
from src.save_experiment_source.i_log_training_source import ILogTrainingSource

with description(ValidationModel, "unit") as self:
    with before.all:
        self.log_sources = [mock(ILogTrainingSource)]
        self.temp_location = "spec/temp/"
        try:
            os.mkdir(self.temp_location)
        except FileExistsError:
            pass

    with after.all:
        shutil.rmtree(self.temp_location)
        unstub()

    with it("Can be initialised with logging source"):
        # Act
        model = ValidationModel(log_sources=self.log_sources)
        # Assert
        expect(model.log_sources).to(equal(self.log_sources))

    with it("Can train model"):
        # Arrange
        model = ValidationModel(log_sources=self.log_sources)
        data = DataFrame({"a": [1, 2, 3, 4]})
        # Act
        training_results = model.train(data)
        # Assert
        expected_results = {"Accuracy": 45, "Error": 8}  # Validation method not defined by input
        expect(training_results).to(equal(expected_results))

    with it("Can test"):
        # Arrange
        model = ValidationModel(log_sources=self.log_sources)
        data = DataFrame({"a": [1, 2, 3, 4]})
        # Act
        training_results = model.test(data)
        # Assert
        expected_results = {"Accuracy": 42, "Error": 9}  # Validation method not defined by input
        expect(training_results).to(equal(expected_results))

    with it("Can save"):
        # Arrange
        path = f"{self.temp_location}model_validation.pkl"
        model = ValidationModel(log_sources=self.log_sources, name="validation")
        # Act
        model.save(self.temp_location)
        # Assert
        expect(os.path.isfile(path)).to(be_true)

    with it("Can load"):
        # Arrange
        path = self.temp_location + "model_loaded.pkl"
        model = ValidationModel(log_sources=self.log_sources, name="loaded")
        model.save(self.temp_location)
        # Act
        model.load(self.temp_location)
        # Assert
        expect(os.path.isfile(path)).to(be_true)
        expect(model.model_loaded_contents).to(equal("Validation model. Mock model saving."))
