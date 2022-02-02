from genpipes import compose
from spec.utils.test_data import mock_data, test_data
from src.pipelines import market_insight_processing as p


def create_mock_pipeline():
    return compose.Pipeline(
        # fmt: off
            steps=[
                ("load test data", test_data, {}),
                ("convert date columns to date_time format", p.convert_date_to_datetime, {}),
                ("filter out data from early 2018", p.filter_column, {"column": "date", "value": "2018-12-01"}),
                ("drop uninteresting colums", p.drop_columns, {"columns": ["root_cat_id"]}),
            ]
    )
