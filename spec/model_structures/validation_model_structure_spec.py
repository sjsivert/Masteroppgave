from pathlib import Path

from expects import be_true, equal, expect
from genpipes.compose import Pipeline
from mamba import before, description, it, after
from mockito.mocking import mock
from mockito.mockito import when, verify, unstub
from mockito.matchers import ANY
from pandas.core.frame import DataFrame
import matplotlib.pyplot as plt

from src.data_types.validation_model import ValidationModel
from src.model_strutures.validation_model_structure import ValidationModelStructure
from src.save_experiment_source.i_log_training_source import ILogTrainingSource

with description(ValidationModelStructure, "unit") as self:
    with before.all:
        self.log_source = mock(ILogTrainingSource)

    with after.all:
        unstub()

    with it("Can be initialised with logging source"):
        # Act
        model_struct = ValidationModelStructure(log_sources=[self.log_source])
        # Assert
        expect(model_struct.log_sources).to(equal([self.log_source]))

    with it("Can process data"):
        # Arrange
        model_struct = ValidationModelStructure([self.log_source])
        pipeline = mock(Pipeline)
        df = DataFrame({"a": [1, 2, 3]})
        when(pipeline).run().thenReturn(df)
        # Act, Assert
        expect(model_struct.process_data(pipeline).equals(df)).to(be_true)

    with it("Can get models"):
        # Arrange
        model = mock(ValidationModel)
        model_struct = ValidationModelStructure([self.log_source])
        model_struct.models = [model]
        # Act, Assert
        expect(model_struct.get_models()).to(equal([model]))

    with it("Can visualize"):
        # Arrange
        model = mock(ValidationModel)
        model_struct = ValidationModelStructure([self.log_source])
        model_struct.models = [model]
        when(model).visualize(ANY)
        # Act
        model_struct.visualize()
        # Assert
        verify(model_struct.models[0], times=1).visualize("Validation_nr_1")

    with it("Can get figures"):
        # Arrange
        model = mock(ValidationModel)
        model_struct = ValidationModelStructure([self.log_source])
        model_struct.models = [model]
        fig_list = [plt.figure(num="Test1"), plt.figure(num="Test2")]
        when(model).get_figures().thenReturn(fig_list)
        # Act
        returned_figures = model_struct.get_figures()
        # Assert
        expect(returned_figures).to(equal(fig_list))
        verify(model, times=1).get_figures()

    with it("Can load models"):
        # Arrange
        model = mock(ValidationModel)
        path = Path("test")
        model_struct = ValidationModelStructure([self.log_source])
        when(ValidationModel).load(ANY).thenReturn(model)
        # Act
        loaded_models = model_struct.load_models([path])
        # Assert
        expect(model_struct.models).to(equal([model]))
        verify(ValidationModel, times=1).load(path)
