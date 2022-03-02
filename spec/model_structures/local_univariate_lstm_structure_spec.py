from unittest.mock import Mock
import matplotlib.pyplot as plt

from expects import be_true, equal, expect
from genpipes.compose import Pipeline
from mamba import description, before, after, shared_context, it, included_context
from mockito import unstub, mock, when, verify
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from src.data_types.lstm_model import LstmModel
from src.pipelines import market_insight_processing as p, local_univariate_lstm_pipeline
from src.pipelines import local_univariate_lstm_pipeline as lstm_pipeline

from spec.mock_config import init_mock_config
from spec.utils.mock_pipeline import create_mock_pipeline
from src.model_strutures.local_univariate_lstm_structure import LocalUnivariateLstmStructure
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from src.utils.config_parser import config

with description(LocalUnivariateLstmStructure, "unit") as self:
    with before.all:
        init_mock_config()
        self.options = config["model"].get()
        self.options["model_structure"] = "local_univariate_lstm"

    with after.each:
        unstub()

    with shared_context("mocks"):
        log_source = mock(ILogTrainingSource)
        log_source.log_pipeline_steps = mock()
        # Mock Lstm Model and methods
        mock_model = Mock(LstmModel)
        when(mock_model).train().thenReturn({})
        when(mock_model).test().thenReturn({})
        fig_list = [plt.figure(num="Test1"), plt.figure(num="Test2")]
        when(mock_model).get_figures().thenReturn(fig_list)

    with it("can be initialised with options"):
        with included_context("mocks"):
            model = LocalUnivariateLstmStructure(
                log_sources=[log_source], **self.options["local_univariate_lstm"]
            )

    with it("can process data"):
        with included_context("mocks"):
            pipeline = create_mock_pipeline()
            model_structure = LocalUnivariateLstmStructure(
                log_sources=[log_source], **self.options["local_univariate_lstm"]
            )
            pipeline_scale_data = Pipeline(
                steps=[
                    (
                        "choose columns 'product_id' and 'date'",
                        p.choose_columns,
                        {"columns": ["date", "product_id"]},
                    ),
                    ("fill in dates with zero values", p.fill_in_dates, {}),
                    ("scale data", p.scale_data, {}),
                    (
                        "split up into train and test data",
                        p.split_into_training_and_test_set,
                        {"training_size": 0.8},
                    ),
                ]
            )
            when(Pipeline, strict=False).run().thenReturn(
                (mock(DataLoader), mock(DataLoader), mock(DataLoader), mock(MinMaxScaler))
            )

            model_structure.init_models()

            model_structure.process_data(data_pipeline=pipeline)

    with it("can train model"):

        with included_context("mocks"):
            model_structure = LocalUnivariateLstmStructure(
                log_sources=[log_source], **self.options["local_univariate_lstm"]
            )
            model_structure.models = [mock_model]
            model_structure.train()
            verify(mock_model, times=1).train()

    with it("can test models"):
        # Mock LSTM model and test method
        with included_context("mocks"):
            model_structure = LocalUnivariateLstmStructure(
                log_sources=[log_source], **self.options["local_univariate_lstm"]
            )
            model_structure.models = [mock_model]
            model_structure.test()
            verify(mock_model, times=1).test()

    with it("can fetch figures"):
        # Mock LSTM model and fet_figures method
        with included_context("mocks"):
            model_structure = LocalUnivariateLstmStructure(
                log_sources=[log_source], **self.options["local_univariate_lstm"]
            )
            model_structure.models = [mock_model]
            fetched_fig_list = model_structure.get_figures()
            expect(fetched_fig_list).to(equal(fig_list))
            verify(mock_model, times=1).get_figures()
