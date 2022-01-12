from expects import be_true, equal, expect
from genpipes.compose import Pipeline
from mamba import before, description, it
from mockito.mocking import mock
from mockito.mockito import when
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

    with it("Can process data"):
        order = 1
        model = LocalUnivariateArimaStructure(order, self.options)
        pipeline = mock(Pipeline)
        df = DataFrame({"a": [1, 2, 3]})
        when(pipeline).run().thenReturn(df)
        expect(model.process_data(pipeline).equals(df)).to(be_true)
