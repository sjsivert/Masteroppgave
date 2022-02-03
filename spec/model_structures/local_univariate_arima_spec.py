from expects import be_true, equal, expect
from genpipes.compose import Pipeline
from mamba import _it, after, before, description, it
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

    with it("Can be initialised with options"):
        order = 1
        log_source = mock(ILogTrainingSource)
        model = LocalUnivariateArimaStructure(
            log_sources=[log_source], **self.options["local_univariate_arima"]
        )

    with _it("Can process data"):
        # TODO: Fix this test
        order = 1
        model = LocalUnivariateArimaStructure(order, **self.options["local_univariate_arima"])
        pipeline = mock(Pipeline)
        pipeline.steps = []
        df = DataFrame({"a": [1, 2, 3]})
        when(pipeline).run().thenReturn(df)
        model.process_data(pipeline)
        # when(LocalUnivariateArimaStructure, strict=False).local_univariate_arima().thenReturn(pipeline)
        verify(pipeline).run()
