import pandas as pd
from expects import be_true, expect
from genpipes import compose, declare
from mamba import description, it

from src.pipelines import market_insight_processing as p

data = [
    {
        "id_x": 0,
        "product_id": 34817620,
        "manufacturer_id": 211757,
        "cat_id": 722,
        "root_cat_id": 11573,
        "id_y": "internettkabel",
        "date": "2021-11-29T04:01:40.409Z",
    },
    {
        "id_x": 1,
        "product_id": 34796949,
        "manufacturer_id": 211757,
        "cat_id": 722,
        "root_cat_id": 11573,
        "id_y": "internettkabel",
        "date": "2021-11-29T04:01:40.409Z",
    },
    {
        "id_x": 2,
        "product_id": 34763798,
        "manufacturer_id": 211757,
        "cat_id": 722,
        "root_cat_id": 11573,
        "id_y": "internettkabel",
        "date": "2021-11-29T04:01:40.409Z ",
    },
]


@declare.generator()
def test_data() -> pd.DataFrame:
    return pd.DataFrame(data)


with description("Market insight prosessing pipeline", "unit") as self:
    with it("It can load data"):
        result = compose.Pipeline(steps=[("load data", test_data, {})]).run()
        expect(result.equals(pd.DataFrame(data))).to(be_true)

    with it("Can run all processing steps in a complete pipeline"):
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
