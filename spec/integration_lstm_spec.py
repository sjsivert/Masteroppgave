# fmt: off
import os
import shutil

from click.testing import CliRunner
from expects import be_true, equal, expect
from expects.matchers.built_in import be
from genpipes.compose import Pipeline
from mamba import _it, after, before, description, it
from mockito import ANY, mock, when
from mockito.mockito import unstub
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from src import main
from src.datasets.time_series_dataset import TimeseriesDataset
from src.pipelines import local_univariate_arima_pipeline as arima_pipeline
from src.pipelines import market_insight_preprocessing_pipeline as pipeline
from src.pipelines import market_insight_processing as market_processing
from src.pipelines import local_univariate_lstm_pipeline as lstm_pipeline
from src.save_experiment_source.local_checkpoint_save_source import LocalCheckpointSaveSource
from src.utils.config_parser import config

from spec.mock_config import init_mock_config
from spec.test_logger import init_test_logging
from spec.utils.mock_pipeline import create_mock_pipeline
from spec.utils.test_data import random_data_loader

with description("main Local Univariate LSTM integration test", "integration") as self:
    with before.all:
        self.runner = CliRunner()
        self.model_struct_type = "local_univariate_lstm"
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
        self.mocked_pipeline = mock(Pipeline)
        self.mocked_pipeline.steps = []

        test_pipeline = Pipeline(
            steps=[("load random generated test data", random_data_loader, {})]
        )
        # fmt: off
        test_local_univariate_pipeline = Pipeline(
            steps=test_pipeline.steps + [
                ("choose columns date and interest", market_processing.choose_columns, {"columns": ["date", "interest"]}),
                ("fill inn dates", market_processing.fill_in_dates, {}),
                ("convert to np.array", market_processing.convert_to_np_array, {}),
                ("scale data", market_processing.scale_data, {}),
                ("split into test and training data", market_processing.split_into_training_and_test_set,
                 {"training_size": 0.8},),
                ("split into validationa nd training data", market_processing.split_into_training_and_validation_set,
                 {"training_size": 0.8},),
                ("convert to timeseries dataset", market_processing.convert_to_time_series_dataset,
                 {"input_window_size": 1, "output_window_size": 1}),
            ]
        )
        train_set, val_set, test_set, scaler = test_local_univariate_pipeline.run()
        when(pipeline).market_insight_pipeline().thenReturn(test_pipeline)
        when(lstm_pipeline).local_univariate_lstm_pipeline(
            data_set=ANY,
            cat_id=ANY,
            training_size=ANY,
            input_window_size=ANY,
            output_window_size=ANY,

        ).thenReturn(self.mocked_pipeline)
        when(self.mocked_pipeline).run().thenReturn((train_set, val_set, test_set, mock(MinMaxScaler)))


    with after.each:
        unstub()

    with after.all:
        shutil.rmtree(self.model_save_location)
        shutil.rmtree(self.checkpoints_location)


    with it("runs with --experiment --save"):
        exp_name = "save_test"

        # Act
        result = self.runner.invoke(
            main.main, ["--experiment", exp_name, "description", "--save"], catch_exceptions=False
        )

        # Assert
        # TODO: Uncomment tests when features are implemented
        expect(result.exit_code).to(be(0))
        # Assert the correct number of files are created after a saved experiment
        expect(os.path.isdir(f"{self.model_save_location}/{exp_name}")).to(be_true)
        #expect(len(os.listdir(f"{self.model_save_location}/{exp_name}/figures"))).to(equal(3))
        expect(os.path.isdir(self.checkpoints_location)).to(be_true)
        expect(os.path.isfile(f"{self.checkpoints_location}/options.yaml")).to(be_true)
        expect(os.path.isfile(f"{self.checkpoints_location}/title-description.txt")).to(be_true)
        if os.path.isfile(f"{self.model_save_location}/{exp_name}/log_file.log"):
            #expect(len(os.listdir(f"{self.model_save_location}/{exp_name}"))).to(equal(9))
            pass
        else:
            pass
            #expect(len(os.listdir(f"{self.model_save_location}/{exp_name}"))).to(equal(8))

    with it("runs with --experiment --tuning"):
        exp_name = "save_tune"

        # Act
        result = self.runner.invoke(
            main.main, ["--experiment", exp_name, "description", "--tune"], catch_exceptions=False
        )

        # Assert
        # TODO: Uncomment tests when features are implemented
        expect(result.exit_code).to(be(0))
        # Assert the correct number of files are created after a saved experiment
        expect(os.path.isdir(f"{self.model_save_location}/{exp_name}")).to(be_true)
        #expect(len(os.listdir(f"{self.model_save_location}/{exp_name}/figures"))).to(equal(3))
        expect(os.path.isdir(self.checkpoints_location)).to(be_true)
        expect(os.path.isfile(f"{self.checkpoints_location}/options.yaml")).to(be_true)
        expect(os.path.isfile(f"{self.checkpoints_location}/title-description.txt")).to(be_true)
        if os.path.isfile(f"{self.model_save_location}/{exp_name}/log_file.log"):
            #expect(len(os.listdir(f"{self.model_save_location}/{exp_name}"))).to(equal(9))
            pass
        else:
            pass
            #expect(len(os.listdir(f"{self.model_save_location}/{exp_name}"))).to(equal(8))

    with it("can continue tunee"):
        exp_name = "continue_tune"

        # Act
        self.runner.invoke(
        main.main, ["--experiment", exp_name, "description", "--tune"], catch_exceptions=False
        )
        result = self.runner.invoke(
            main.main, ["-c", "--tune"], catch_exceptions=False
        )
