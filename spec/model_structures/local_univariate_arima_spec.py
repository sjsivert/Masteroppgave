from collections import OrderedDict

from expects import be_true, equal, expect, match
from genpipes.compose import Pipeline
from mamba import _it, after, before, description, it, shared_context, included_context
from mockito.mocking import mock
from mockito.mockito import unstub, verify, when
from pandas.core.frame import DataFrame
from spec.mock_config import init_mock_config
from src.model_strutures.local_univariate_arima_structure import LocalUnivariateArimaStructure
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from src.utils.config_parser import config

with description(LocalUnivariateArimaStructure, "unit") as self:
    with before.all:
        init_mock_config()
        self.options = config["model"].get()

    with after.each:
        unstub()

    with shared_context("mocks"):
        log_source = mock(ILogTrainingSource)
    with it("Can be initialised with options"):
        with included_context("mocks"):
            model = LocalUnivariateArimaStructure(
                log_sources=[log_source], **self.options["local_univariate_arima"]
            )

    with _it("Can process data"):
        with included_context("mocks"):
            # todo: fix this test
            pipeline = mock(Pipeline)
            pipeline.steps = []
            df = DataFrame({"a": [1, 2, 3]})
            when(pipeline).run().thenReturn((df, df))
            model.process_data(pipeline)
            # when(LocalUnivariateArimaStructure, strict=False).local_univariate_arima().thenReturn(pipeline)
            verify(pipeline).run()

    with it("autotunes with correct values fetched from the config"):
        with included_context("mocks"):
            tuning_options = OrderedDict([("p", (1, 2)), ("d", (3, 4)), ("q", (5, 6))])
            model = LocalUnivariateArimaStructure(
                log_sources=[log_source],
                **self.options["local_univariate_arima"],
                hyperparameter_tuning_range=tuning_options
            )

            expect(model.hyperparameter_tuning_range.__str__()).to(match(tuning_options.__str__()))
