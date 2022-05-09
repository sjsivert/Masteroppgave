# fmt: off
from genpipes.compose import Pipeline
from pandas import DataFrame
from src.pipelines import market_insight_processing as market_processing
from src.pipelines.data_loader import dataframe_to_generator


def arima_model_pipeline(data_set: DataFrame, cat_id: str, forecast_size: int) -> Pipeline:
    """
    Datapipeline which processes the data to be on the correct format
    before the local_univariate_arima model can be applied.

    Args:
        :param pipeline:
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
            (f"...", market_processing.dont_scale, {}),
            (f"split up into training set and test set of forecast window size{forecast_size})",
             market_processing.split_into_training_and_test_forecast_window_arima,
                {"forecast_window_size": forecast_size, "input_window_size": 0}),
            (f"split up training set into training and  validation set of forecast window size{forecast_size})",
             market_processing.split_into_training_and_validation_forecast_window_arima,
             {"forecast_window_size": forecast_size, "input_window_size": 0}),
             #("save datasets to file", market_processing.save_datasets_to_file_arima, {"name": "scaled"}),
        ]
    )
