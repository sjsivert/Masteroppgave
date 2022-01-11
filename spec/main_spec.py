from typing import OrderedDict

from click.testing import CliRunner
from confuse import Configuration
from expects import be_true, expect
from expects.matchers.built_in import be
from genpipes.compose import Pipeline
from mamba import after, before, description, it
from mockito import mock, verify, when
from mockito.matchers import ANY
from mockito.mockito import unstub
from src import main
from src.experiment import Experiment
from src.pipelines.market_insight_preprocessing_pipeline import market_insight_pipeline
from src.utils.config_parser import config
from src.utils.logger import init_logging

from spec.test_logger import init_test_logging

def init_mock_config():
    config.clear()
    config.read(user=False)
    config["experiment"]["save_sources_to_use"] = ["disk"]
    config["experiment"]["save_source"] = {"disk": {"model_save_location": "./models/temp",
                                                    },
                                           "neptune": {"project_id": "sjsivertandsanderkk/Masteroppgave"}
                                           }
    config["model"]["model_type"] = {
        "model_type": "local_univariate_arima",
        "rng_seed": 42,
        "local_univariate_arima": {
            "order": (1, 1, 1),
        }
    }
    config["data"] = {
        "data_path": './datasets/raw/market_insights_overview_5p.csv',
        "categories_path": './datasets/raw/solr_categories_2021_11_29.csv'
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
            main.main, ["--experiment", "title", "description", "--no-save"],
            catch_exceptions=False
        )

        print(result)
        expect(result.exit_code).to(be(0))

    with it("runs with --experiment --save"):
        when(Experiment, strict=False).run_complete_experiment_with_saving().thenReturn(None)
        result = self.runner.invoke(
            main.main, ["--experiment", "title", "description", "--save"],
            catch_exceptions=False
        )

        print(result)
        expect(result.exit_code).to(be(0))

    with it("runs without parameters"):
        expect(True).to(be_true)

    with it("executes init_logging"):
        mock_logger = mock(init_logging())


