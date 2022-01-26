import pandas as pd
from expects import be_true, expect
from genpipes import compose
from mamba import description, it, before

from spec.test_logger import init_test_logging
from spec.utils.test_data import test_data, mock_data
from src.pipelines import market_insight_processing as p

with description("Market insight prosessing pipeline", "unit") as self:
    with before.all:
        init_test_logging()

    with it("It can load data"):
        # noinspection PyTypeChecker
        result = compose.Pipeline(steps=[("load data", test_data, {})]).run()
        expect(result.equals(pd.DataFrame(mock_data))).to(be_true)

    with it("Can run all processing steps in a complete pipeline"):
        # noinspection PyTypeChecker
        result = compose.Pipeline(
            steps=[
                ("load data", test_data, {}),
                ("convert date columns to date_time format", p.convert_date_to_datetime, {}),
                ("sum up clicks to category level", p.group_by, {"group_by": ["date", "cat_id"]}),
                (
                    "filter out data from early 2018",
                    p.filter_column,
                    {"column": "date", "value": "2018-12-01"},
                ),
                ("drop uninteresting colums", p.drop_columns, {"columns": ["root_cat_id"]}),
            ]
        ).run()
        expect(result.equals(result)).to(be_true)
