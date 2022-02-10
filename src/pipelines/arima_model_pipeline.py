# fmt: off
from genpipes.compose import Pipeline
from pandas import DataFrame

from src.pipelines import market_insight_processing as market_processing
from src.pipelines.data_loader import dataframe_to_generator


def arima_model_pipeline(data_set: DataFrame, cat_id: str, training_size: float) -> Pipeline:
    """
    Datapipeline which processes the data to be on the correct format
    before the local_univariate_arima model can be applied.

    Args:
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
            ("choose columns 'hits' and 'date'", market_processing.choose_columns, {"columns": ["date", "hits"]}),
            ("fill in dates with zero values", market_processing.fill_in_dates, {}),
            ("split up into training set ({training_size}) and test set ({1 - training_size})",
             market_processing.split_into_training_and_test_set,
                {"training_size": training_size}),
        ]
    )
