from expects import be_true, equal, expect
from genpipes.compose import Pipeline
from mamba import before, description, it, _it
from mockito.mocking import mock
from mockito.mockito import when, verify
from pandas.core.frame import DataFrame

from src.model_strutures.local_univariate_arima_structure import LocalUnivariateArimaStructure

with description(LocalUnivariateArimaStructure, "unit") as self:
    with before.all:
        self.options = {
            "model_type": "local_univariate_arima",
            "local_univariate_arima": {"order": (1, 1, 1)},
        }

    with it("Can be initialised with options"):
        order = 1
        model = LocalUnivariateArimaStructure(order, **self.options["local_univariate_arima"])
        expect(model.order).to(equal(self.options["local_univariate_arima"]["order"]))

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
