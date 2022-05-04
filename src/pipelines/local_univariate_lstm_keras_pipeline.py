# fmt: off
import src.pipelines.market_insight_processing as market_processing
from genpipes.compose import Pipeline
from pandas import DataFrame
from src.pipelines.data_loader import dataframe_to_generator
from src.pipelines.date_feature_generator import (calculate_day_of_the_week,
                                                  calculate_season)


def local_univariate_lstm_keras_pipeline(
        data_set: DataFrame,
        cat_id: str,
        input_window_size: int,
        output_window_size: int,
) -> Pipeline:
    """
    Datapipeline which processes the data to be on the correct format
    before the local_univariate_lstm model can be applied.

    Args:
        :param batch_size:
        :param output_window_size:
        :param input_window_size:
        :param cat_id:
        :param data_set:
        :param pipeline:
        :param training_size:
    Returns:
        Pipeline: The pipeline with the steps added.
    """
    return Pipeline(
        steps=[
            ("Convert input dataset to generator object", dataframe_to_generator, {"df": data_set}),
            (f"filter out category {cat_id})",
                market_processing.filter_by_cat_id,
                {"cat_id": cat_id},),
            ("choose columns 'interest' and 'date'", market_processing.choose_columns, {"columns": ["date", "interest"]}),
            ("fill in dates with zero values", market_processing.fill_in_dates, {}),
            ("convert to np.array", market_processing.convert_to_np_array_univariate, {}),
            ("scale down outliers", market_processing.scale_down_outliers, {}),
            ("decrease variance by taking the log", market_processing.decrease_variance, {}),
            (f"Normalize the data ", market_processing.scale_data, {"should_scale": True}),
            (f"Remove trend and seasons by differencing the series t[i] - t[i-1]", market_processing.differencing, {}),
            (f"generate x y pairs with sliding window with input size {input_window_size}, and output size {output_window_size}",
                market_processing.sliding_window_x_y_generator,
                {"input_window_size": input_window_size, "output_window_size": output_window_size}),
            (f"generate training and test data with test data size {output_window_size}, and training data = sequence_length - {output_window_size}",
                market_processing.keras_split_into_training_and_test_set, {"test_window_size": output_window_size}),

            #(f"split up into training set ({training_size}) and test set ({1 - training_size})",
                #market_processing.split_into_training_and_test_forecast_window,
                #{"input_window_size": input_window_size, "forecast_window_size": output_window_size}),
            #(f"split training set into train set with val window of {output_window_size}) ",
                #market_processing.split_into_training_and_validation_forecast_window,
                #{"input_window_size": input_window_size, "forecast_window_size": output_window_size}),
            #(f"convert to timeseries dataset with input window size of {input_window_size}, "
                #f"and output window size of {output_window_size}", market_processing.convert_to_time_series_dataset,
                #{"input_window_size": input_window_size, "output_window_size": output_window_size}),
        ]
    )
