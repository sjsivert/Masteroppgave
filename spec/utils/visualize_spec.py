from expects import be_true, expect
from mamba import description, it
from src.utils.error_calculations import *
from src.utils.visuals import visualize_data_series
from matplotlib.figure import Figure
from pandas import DataFrame


with description("visualize figures", "unit") as self:

    with it("can visualize one plot"):
        data_set_1 = DataFrame([1, 2, 3, 4, 5])
        data_set_2 = DataFrame([1, 2, 3, 4, 5])
        fig = visualize_data_series(
            title="Uni_test",
            data_series=[data_set_1],
            data_labels=["uni_test_series"],
            colors=["blue"],
            x_label="date",
            y_label="error",
        )
        expect(isinstance(fig, Figure)).to(be_true)

    with it("can visualize multiple plots"):
        data_set_1 = DataFrame([1, 2, 3, 4, 5])
        data_set_2 = DataFrame([1, 2, 3, 4, 5])
        fig = visualize_data_series(
            title="Multi_test",
            data_series=[data_set_1, data_set_2],
            data_labels=["multi_test_series_1", "multi_test_series_2"],
            colors=["blue", "orange"],
            x_label="date",
            y_label="error",
        )
        expect(isinstance(fig, Figure)).to(be_true)
