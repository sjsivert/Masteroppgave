import os
import shutil

from click.testing import CliRunner
from confuse import Configuration
from expects import be_true, expect, equal
from expects.matchers.built_in import be
from genpipes.compose import Pipeline
from mamba import after, before, description, it
from mockito import mock, when
from mockito.mockito import unstub

from spec.mock_config import init_mock_config
from spec.test_logger import init_test_logging
from src import main
from src.pipelines import market_insight_preprocessing_pipeline as pipeline
from src.save_experiment_source.local_checkpoint_save_source import LocalCheckpointSaveSource
from src.utils.config_parser import config

with description("main integration test", "api") as self:
    with before.all:
        self.runner = CliRunner()
        self.model_struct_type = "validation_model"
        self.model_save_location = "./models/temp"
        try:
            os.mkdir(self.model_save_location)
        except FileExistsError:
            pass
        self.checkpoints_location = LocalCheckpointSaveSource().get_checkpoint_save_location()
        init_test_logging()

    with before.each:
        init_mock_config(self.model_struct_type, self.model_save_location)
        # Mock pipelines
        mocked_pipeline = mock(Pipeline)
        when(mocked_pipeline).run()
        when(pipeline).market_insight_pipeline().thenReturn(mocked_pipeline)

    with after.each:
        unstub()

    with after.all:
        shutil.rmtree(self.model_save_location)
        shutil.rmtree(self.checkpoints_location)

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
        expect(len(os.listdir(f"{self.model_save_location}/{exp_name}"))).to(equal(8))
        expect(len(os.listdir(f"{self.model_save_location}/{exp_name}/figures"))).to(equal(3))
        expect(os.path.isdir(self.checkpoints_location)).to(be_true)
        expect(os.path.isfile(f"{self.checkpoints_location}/options.yaml")).to(be_true)
        expect(os.path.isfile(f"{self.checkpoints_location}/title-description.txt")).to(be_true)

    with it("can continue the last ran experiment from disk"):
        # Add neptune as save_source for this experiment
        old_conf = config["experiment"].get()
        new_conf = old_conf.copy()
        new_conf["save_sources_to_use"] = ["disk", "neptune"]
        config["experiment"].set(new_conf)

        # Run the experiment once
        self.runner.invoke(
            main.main,
            ["--experiment", "test-continue-exp", "description", "--save"],
            catch_exceptions=False,
        )
        # change config
        old_conf = config["experiment"].get()
        new_conf = old_conf.copy()
        new_conf["tags"] = ["test-continue-exp"]
        config["experiment"].set(new_conf)

        expect(os.path.isdir(f"{self.model_save_location}/test-continue-exp")).to(be_true)
        expect(os.path.isdir(f"{self.checkpoints_location}")).to(be_true)
        expect(old_conf).to_not(equal(new_conf))

        # Continue the first experiment which should use the original config
        self.runner.invoke(main.main, ["--continue"], catch_exceptions=False)
        saved_config = Configuration("config", __name__)
        saved_config.set_file(f"{self.model_save_location}/test-continue-exp/options.yaml")
        # expect the saved config to match the original config used to run the experiment
        # and not the changed config
        expect(saved_config.dump()).to_not(equal(config.dump()))

    with it("can load a previously saved experiment"):
        exp_name = "test-load-experiment"
        self.runner.invoke(
            main.main, ["--experiment", exp_name, "description", "--save"], catch_exceptions=False
        )
        expect(os.path.isdir(f"{self.model_save_location}/{exp_name}")).to(be_true)

        self.runner.invoke(
            main.main, ["--load", f"{self.model_save_location}/{exp_name}"], catch_exceptions=False
        )
