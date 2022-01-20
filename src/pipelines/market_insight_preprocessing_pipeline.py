from genpipes import compose

from src.pipelines import data_loader
from src.pipelines import market_insight_processing as p
from src.utils.config_parser import config, get_absolute_path

columns_to_drop = [
    "internal_doc_id",
    "id_x",
    "manufacturer_id",
    "_version_",
    "internal_doc_id",
    "id_y",
    "adult",
]


# fmt: off
def market_insight_pipeline() -> compose.Pipeline:
    pipeline = compose.Pipeline(steps=[
        ("load market insight data and categories and merge them",
         data_loader.load_and_merge_market_insight_and_categories,
         {"market_insight_path": get_absolute_path(config['data']['data_path'].get()),
          "categories_path": get_absolute_path(config['data']['categories_path'].get())}),
        ("convert date columns to date_time format", p.convert_date_to_datetime, {}),
        ("sum up clicks to category level [groupBy(date, cat_id)]", p.group_by, {"group_by": ["date", "cat_id"]}),
        ("filter out data from early 2018-12-01", p.filter_column, {"column": "date", "value": "2018-12-01"}),
        (f"drop uninteresting colums: {columns_to_drop.__str__()}", p.drop_columns,
         {"columns": columns_to_drop}),
    ])

    return pipeline
