from click.testing import CliRunner
from expects import be_true, expect
from expects.matchers.built_in import be
from mamba import after, before, description, it
from mockito import mock, when
from mockito.mockito import unstub

from spec.test_logger import init_test_logging
from src import main
from src.experiment import Experiment
from src.utils.config_parser import config
from src.utils.logger import init_logging


def init_mock_config():
    config.clear()
    config.read(user=False)
    config["experiment"]["save_sources_to_use"] = []
    config["experiment"]["save_source"] = {
        "disk": {
            "model_save_location": "./models/temp",
        },
        "neptune": {"project_id": "sjsivertandsanderkk/Masteroppgave"},
    }
    config["model"]["model_type"] = {
        "model_type": "local_univariate_arima",
        "rng_seed": 42,
        "local_univariate_arima": {
            "order": (1, 1, 1),
        },
    }
    config["data"] = {
        "data_path": "./datasets/raw/market_insights_overview_5p.csv",
        "categories_path": "./datasets/raw/solr_categories_2021_11_29.csv",
    }


with description("main.py", "integration") as self:
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
        when(Experiment, strict=False).run_complete_experiment_without_saving().thenReturn(None)
        result = self.runner.invoke(
            main.main, ["--experiment", "title", "description", "--no-save"], catch_exceptions=False
        )

        expect(result.exit_code).to(be(0))

    with it("runs with --experiment --save"):
        when(Experiment, strict=False).run_complete_experiment_with_saving().thenReturn(None)

        result = self.runner.invoke(
            main.main, ["--experiment", "title", "description", "--save"], catch_exceptions=False
        )

        expect(result.exit_code).to(be(0))

    with it("runs without parameters"):
        expect(True).to(be_true)

    with it("executes init_logging"):
        mock_logger = mock(init_logging())
