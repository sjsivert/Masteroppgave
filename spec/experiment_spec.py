import os
import shutil

import expects
import pytest
from confuse import Configuration
from expects import be, be_above, expect
from genpipes.compose import Pipeline
from mamba import after, before, description, included_context, it, shared_context
from mockito import mock, unstub, when
from mockito.mockito import verify
from pandas.core.frame import DataFrame

from spec.test_logger import init_test_logging
from src.experiment import Experiment
from src.model_strutures.i_model_structure import IModelStructure
from src.model_strutures.local_univariate_arima_structure import LocalUnivariateArimaStructure
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
        model = mock(IModelStructure)
        df = DataFrame({"a": [1, 2, 3]})
        when(model).process_data(pipeline).thenReturn(df)
        experiment.model_structure = model

        dataframe = experiment._load_and_process_data(pipeline)
        expect(dataframe).to(be(df))

    with it("can choose_model_structure arima"):
        experiment = Experiment("title", "description")
        options = {
            "model_type": "local_univariate_arima",
            "local_univariate_arima": {"order": (1, 1, 1)},
        }
        experiment._choose_model_structure(options)
        expect(experiment.model_structure).to_not(expects.be_none)

    with it("can choose_model_structure validation model"):
        experiment = Experiment("title", "description")
        options = {
            "model_type": "validation_model",
        }
        experiment._choose_model_structure(options)
        expect(experiment.model_structure).to_not(expects.be_none)

    with it("raise exception when wrong model structure is chosen"):
        experiment = Experiment("title", "description")
        options = {
            "model_type": "wrong_model_structure",
            "wrong_model_structure": {"order": (1, 1, 1)},
        }
        with pytest.raises(KeyError):
            experiment._choose_model_structure(options)

    with it("can train model"):
        # Arrange
        experiment = Experiment("title", "description")
        experiment.model_structure = mock(LocalUnivariateArimaStructure)
        when(experiment.model_structure).train()
        # Act
        experiment._train_model()
        # Assert
        verify(experiment.model_structure, times=1).train()

    with it("can test_model()"):
        # Arrange
        experiment = Experiment("title", "description")
        experiment.model_structure = mock(LocalUnivariateArimaStructure)
        when(experiment.model_structure).test()
        # Act
        experiment._test_model()
        # Assert
        verify(experiment.model_structure, times=1).test()

    with it("can save_model()"):
        # Arrange
        experiment = Experiment("title", "description")
        when(SaveLocalDiskSource, strict=False).save_model_and_metadata()
        # Act
        experiment._save_model({})

    with shared_context("mock private methods context"):
        # Arrange
        pipeline = mock(Pipeline)

        experiment = Experiment("title", "description")
        when(experiment, strict=False)._load_and_process_data().thenReturn(
            DataFrame({"a": [1, 2, 3]})
        )
        when(experiment, strict=False)._choose_model_structure()
        when(experiment)._train_model()
        when(experiment)._test_model()

    with it("can run_complete_experiment() without saving"):
        with included_context("mock private methods context"):
            # Act
            experiment.run_complete_experiment(
                model_options={
                    "model_type": "local_univariate_arima",
                    "local_univariate_arima": {"order": (1, 1, 1)},
                },
                data_pipeline=pipeline,
                save=False,
            )
            # Assert
            verify(experiment, times=1)._load_and_process_data(data_pipeline=pipeline)

    with it("can run_complete_experiment() with saving"):
        # Arrange
        with included_context("mock private methods context"):
            configuration = mock(Configuration)

            when(configuration).dump().thenReturn("")
            # Act
            experiment.run_complete_experiment(
                model_options={
                    "model_type": "local_univariate_arima",
                    "local_univariate_arima": {"order": (1, 1, 1)},
                },
                data_pipeline=pipeline,
                options_to_save=configuration,
            )
            # Assert
            verify(experiment, times=1)._load_and_process_data(data_pipeline=pipeline)
