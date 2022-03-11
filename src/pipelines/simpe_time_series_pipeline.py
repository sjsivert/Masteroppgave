# fmt: off
from genpipes.compose import Pipeline
from pandas import DataFrame

from src.pipelines import market_insight_processing as market_processing
from src.pipelines.data_loader import dataframe_to_generator, load_csv_data


def simple_time_series_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("load data",  load_csv_data, {"path": "datasets/external/Alcohol_Sales.csv"}),
            ("choose columns", market_processing.choose_columns, {"columns": ["S4248SM144NCEN"]}),
            ("Scale data, expand dims, split, train test, val", market_processing.simple_time_series_processor, {}),
        ]
    )
