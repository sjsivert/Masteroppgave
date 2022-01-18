import os
import shutil
from pathlib import Path

from click.testing import CliRunner
from expects import be_true, expect, equal
from expects.matchers.built_in import be
from mamba import after, before, description, it
from mockito import mock, when
from mockito.mockito import unstub, verify

from spec.mock_config import init_mock_config
from spec.test_logger import init_test_logging
from src import main
from src.continue_experiment import ContinueExperiment
from src.experiment import Experiment
from src.utils.config_parser import config
from src.utils.logger import init_logging
from src.pipelines import market_insight_preprocessing_pipeline as pipeline
from genpipes.compose import Pipeline

with description(
    "Integration test to validate data flow with Validation model", "integration"
) as self:
    with before.all:
        self.runner = CliRunner()
        init_test_logging()
        self.model_struct_type = "validation_model"
        self.model_save_location = "./models/temp"
        try:
            os.mkdir(self.model_save_location)
        except FileExistsError:
            pass
        init_mock_config(self.model_struct_type, self.model_save_location)

    with before.each:
        # Mock pipelines
        mocked_pipeline = mock(Pipeline)
        when(mocked_pipeline).run()
        when(pipeline).market_insight_pipeline().thenReturn(mocked_pipeline)

    with after.each:
        unstub()

    with after.all:
        shutil.rmtree(self.model_save_location)

    with it("runs without errors"):
        result = self.runner.invoke(main.main, [])
        expect(result.exit_code).to(be(0))

    with it("runs with --help"):
        result = self.runner.invoke(main.main, ["--help"])
        expect(result.exit_code).to(be(0))

    with it("runs with --experiment --no-save"):
        result = self.runner.invoke(
            main.main, ["--experiment", "title", "description", "--no-save"], catch_exceptions=False
        )
        expect(result.exit_code).to(be(0))

    with it("runs with --experiment --save"):
        exp_name = "save_test"
        result = self.runner.invoke(
            main.main, ["--experiment", exp_name, "description", "--save"], catch_exceptions=False
        )
        expect(result.exit_code).to(be(0))
        # Assert the correct number of files are created after a saved experiment
        expect(os.path.isdir(f"{self.model_save_location}/{exp_name}")).to(be_true)
        expect(len(os.listdir(f"{self.model_save_location}/{exp_name}"))).to(equal(5))
        expect(len(os.listdir(f"{self.model_save_location}/{exp_name}/figures"))).to(equal(2))
