# fmt: off
from typing import List

from genpipes.compose import Pipeline
from pandas import DataFrame

from src.pipelines.data_loader import dataframe_to_generator
import src.pipelines.market_insight_processing as market_processing


def global_univariate_lstm_pipeline(
        data_set: DataFrame,
        cat_ids: List[str],
        input_window_size: int,
        output_window_size: int,
) -> Pipeline:
    """
    Datapipeline which processes the data to be on the correct format
    before the local_univariate_lstm model can be applied.

    Args:
        :param cat_ids:
        :param output_window_size:
        :param input_window_size:
        :param data_set:
    Returns:
        Pipeline: The pipeline with the steps added.
    """
    return Pipeline(
        steps=[
            ("Convert input dataset to generator object", dataframe_to_generator, {"df": data_set}),
            ("split into multiple datasets from the columns", market_processing.global_pipeline_processor,
             {"cat_ids": cat_ids,
              "input_window_size": input_window_size,
              "output_window_size": output_window_size,
              }),
        ]
    )
