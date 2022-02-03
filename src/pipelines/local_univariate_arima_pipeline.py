# fmt: off
from genpipes.compose import Pipeline
from src.pipelines import market_insight_processing as market_processing


def local_univariate_arima_pipeline(pipeline: Pipeline, training_size: float) -> Pipeline:
    """
    Datapipeline which processes the data to be on the correct format
    before the local_univariate_arima model can be applied.

    Args:
        :param pipeline:
        :param training_size:
    Returns:
        Pipeline: The pipeline with the steps added.
    """
    return Pipeline(steps=
                    pipeline.steps + [
                        ("filter out category 'Nettverkskabler' (11573)", market_processing.filter_by_cat_id,
                         {"cat_id": 11573}),
                        ("choose columns 'hits' and 'date'", market_processing.choose_columns,
                         {"columns": ["date", "hits"]}),
                        ("fill in dates with zero values", market_processing.fill_in_dates, {}),
                        (f"split up into training set ({training_size}) and test set ({1 - training_size})",
                         market_processing.split_into_training_and_test_set, {"training_size": training_size}),
                    ])
