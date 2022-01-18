from pathlib import Path

from click.testing import CliRunner
from expects import be_true, expect
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

with description("main.py", "unit") as self:
    with before.all:
        self.runner = CliRunner()
        init_test_logging()

        init_mock_config()

    with after.all:
        unstub()

    with it("runs without errors"):
        result = self.runner.invoke(main.main, [])
        expect(result.exit_code).to(be(0))

    with it("runs with --help"):
        result = self.runner.invoke(main.main, ["--help"])
        expect(result.exit_code).to(be(0))

    with it("runs with --experiment --no-save"):
        when(Experiment, strict=False).run_complete_experiment().thenReturn(None)
        result = self.runner.invoke(
            main.main, ["--experiment", "title", "description", "--no-save"], catch_exceptions=False
        )

        expect(result.exit_code).to(be(0))

    with it("runs with --experiment --save"):
        when(Experiment, strict=False).run_complete_experiment().thenReturn(None)

        result = self.runner.invoke(
            main.main, ["--experiment", "title", "description", "--save"], catch_exceptions=False
        )

        expect(result.exit_code).to(be(0))

    with it("runs without parameters"):
        expect(True).to(be_true)

    with it("executes init_logging"):
        mock_logger = mock(init_logging())

    with it("runs with --continue-experiment"):
        # Arrange
        checkpoint_save_location = Path(
            config["experiment"]["save_source"]["disk"]["checkpoint_save_location"].get()
        )
        mock_experiment = mock(ContinueExperiment)
        when(main).ContinueExperiment(
            experiment_checkpoints_location=checkpoint_save_location
        ).thenReturn(mock_experiment)
        when(mock_experiment).continue_experiment().thenReturn(None)

        # Act
        self.runner.invoke(main.main, ["--continue-experiment"], catch_exceptions=False)

        # Assert
        verify(mock_experiment).continue_experiment()
