import os
import shutil

import expects
import pytest
from expects import be, be_above, expect
from genpipes.compose import Pipeline
from mamba import after, before, description, it
from mockito import mock, unstub, when
from mockito.matchers import ANY
from mockito.mockito import verify
from pandas.core.frame import DataFrame

from spec.test_logger import init_test_logging
from src.experiment import Experiment
from src.model_strutures.i_model_type import IModelType
from src.model_strutures.local_univariate_arima import LocalUnivariateArima
from src.save_experiment_source.save_local_disk_source import SaveLocalDiskSource

with description(Experiment, "integration") as self:

    with before.all:
        init_test_logging()
        self.temp_location = "spec/temp/"
        try:
            os.mkdir(self.temp_location)
        except FileExistsError:
            pass

    with after.all:
        shutil.rmtree(self.temp_location)
        unstub()

    with it("initialises with title and description", "integration"):
        experiment = Experiment("title", "description")
        expect(experiment.title).to(be("title"))
        expect(experiment.experiment_description).to(be("description"))

    with it("initialises with title and description and save source disk", "integration"):
        save_sources = ["disk"]
        save_source_options = {"disk": {"model_save_location": "./spec/temp/"}}
        experiment = Experiment("title", "description", save_sources, save_source_options)

        expect(len(experiment.save_sources)).to(be_above(0))

    with it("returns dataframe on load_and_process_data()", "integration"):
        experiment = Experiment("title", "description")
        pipeline = mock(Pipeline)
        model = mock(IModelType)
        df = DataFrame({"a": [1, 2, 3]})
        when(model).process_data(pipeline).thenReturn(df)
        experiment.model = model

        dataframe = experiment.load_and_process_data(pipeline)
        expect(dataframe).to(be(df))

    with it("can choose_model_structure arima"):
        experiment = Experiment("title", "description")
        options = {
            "model_type": "local_univariate_arima",
            "local_univariate_arima": {"order": (1, 1, 1)},
        }
        experiment.choose_model_structure(options)
        expect(experiment.model).to_not(expects.be_none)

    with it("raise exception when wrong model structure is chosen"):
        experiment = Experiment("title", "description")
        options = {
            "model_type": "wrong_model_structure",
            "wrong_model_structure": {"order": (1, 1, 1)},
        }
        with pytest.raises(KeyError):
            experiment.choose_model_structure(options)

    with it("can train_model()"):
        # Arrange
        experiment = Experiment("title", "description")
        experiment.model = mock(LocalUnivariateArima)
        when(experiment.model).train_model()
        # Act
        experiment.train_model()
        # Assert
        verify(experiment.model, times=1).train_model()

    with it("can test_model()"):
        # Arrange
        experiment = Experiment("title", "description")
        experiment.model = mock(LocalUnivariateArima)
        when(experiment.model).test_model()
        # Act
        experiment.test_model()
        # Assert
        verify(experiment.model, times=1).test_model()

    with it("can save_model()"):
        # Arrange
        save_source = mock(SaveLocalDiskSource)
        experiment = Experiment("title", "description")
        experiment.save_sources = [save_source]
        when(save_source).save_options(ANY)
        when(save_source).save_metrics(ANY)
        # Act
        experiment.save_model({})
        # Assert
        verify(experiment.save_sources[0], times=1).save_options({})
        verify(experiment.save_sources[0], times=1).save_metrics([])
