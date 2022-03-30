from genpipes.compose import Pipeline

from spec.utils.test_data import random_data_loader
from src.pipelines import market_insight_processing as market_processing


def mock_time_series_generator(input_window_size: int, output_window_size: int):
    return Pipeline(
        steps=[
            ("load random generated test data", random_data_loader, {}),
            ("choose colums", market_processing.choose_columns, {"columns": ["date", "interest"]}),
            ("fill inn dates", market_processing.fill_in_dates, {}),
            ("convert to np.array", market_processing.convert_to_np_array, {}),
            ("scale data", market_processing.scale_data, {}),
            (
                "sliding window",
                market_processing.sliding_window_x_y_generator,
                {"input_window_size": input_window_size, "output_window_size": output_window_size},
            ),
        ]
    )
