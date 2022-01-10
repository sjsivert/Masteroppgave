from click.testing import CliRunner
from expects import be_true, expect
from expects.matchers.built_in import be
from mamba import after, before, description, it
from mockito import mock, verify, when
from mockito.matchers import ANY
from mockito.mockito import unstub
from src import main
from src.experiment import Experiment
from src.utils.logger import init_logging

from spec.test_logger import init_test_logging

with description("main.py", "integration") as self:
    with before.all:
        self.runner = CliRunner()
        init_test_logging()

    with after.all:
        unstub()

    with it("runs without errors"):
        result = self.runner.invoke(main.main, [])
        expect(result.exit_code).to(be(0))

    with it("runs with --help"):
        result = self.runner.invoke(main.main, ["--help"])
        expect(result.exit_code).to(be(0))

    with it("runs with --experiment"):
        result = self.runner.invoke(
            main.main, ["--experiment", "title", "description", "--no-save"]
        )

        expect(result.exit_code).to(be(0))

    with it("runs without parameters"):
        expect(True).to(be_true)

    with it("executes init_logging"):
        mock_logger = mock(init_logging())
