from src.utils.config_parser import config

from expects import be_true, equal, expect
from genpipes.compose import Pipeline
from mamba import before, description, it, _it, after
from mockito.mocking import mock
from mockito.mockito import when, verify, unstub
from pandas.core.frame import DataFrame

from src.model_strutures.local_univariate_arima_structure import LocalUnivariateArimaStructure

from spec.mock_config import init_mock_config
from src.save_experiment_source.i_log_training_source import ILogTrainingSource

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
        expect(model.training_size).to(
            equal(self.options["local_univariate_arima"]["training_size"])
        )

    with _it("Can process data"):
        # TODO: Fix this test
        order = 1
        model = LocalUnivariateArimaStructure(order, self.options)
        pipeline = mock(Pipeline)
        pipeline.steps = []
        df = DataFrame({"a": [1, 2, 3]})
        when(pipeline).run().thenReturn(df)
        model.process_data(pipeline)
        # when(LocalUnivariateArimaStructure, strict=False).local_univariate_arima().thenReturn(pipeline)
        verify(pipeline).run()
