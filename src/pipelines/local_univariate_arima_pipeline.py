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
                        ("choose columns 'interest' and 'date' and 'cat_id'", market_processing.choose_columns,
                         {"columns": ["date", "interest", "cat_id"]}),
                        ("fill in dates with zero values", market_processing.fill_in_dates, {}),
                    ])
