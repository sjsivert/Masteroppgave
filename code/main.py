import click
import logging
import pandas as pd
from typing import Iterable
from prefect import task, Flow
from config_parser import config, get_absolute_path
import data.data_loader as data_loader
from genpipes import declare, compose
from features import market_insight_processing as p


# def pipeline():
#   with Flow("Main Pipeline") as flow:
#     file_name = get_absolute_path(config['data']['data_path'].get())
#     data = data_loader.load_data(file_name)
#     return flow

@click.command()
@click.option('--experiment_description', '-e', default='', required=False, help='Experiment description.')
def main(experiment_description):
  config['experiment_description'] = experiment_description
  logging.basicConfig(
    # filename='main.log',
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s {%(module)s} [%(funcName)s] %(message)s',
    datefmt='%Y-%m-%d,%H:%M:%S:%f'
    )

  logging.info('Started')

  # pipeline_flow = pipeline()
  # pipeline_flow.run()

  load_and_process_data_pipeline = compose.Pipeline(steps=[
    ("load market insight data and categories and merge them", data_loader.load_and_merge_market_insight_and_categories, 
      {"market_insight_path": get_absolute_path(config['data']['data_path'].get()), 
        "categories_path": get_absolute_path(config['data']['categories_path'].get())}),
    # ("load market insight data", data_loader.load_csv_data, {"path": config['data']['data_path'].get()}),
    # ("load categories data", data_loader.load_csv_data, {"path": config['data']['categories_path'].get()}),
    # ("merge categories with market insight data", p.merge, {"left_on": "category_id", "right_on": "internal_doc_id"}),
    ("convert date columns to date_time format", p.convert_date_to_datetime, {}),
    ("print data", p.print_df, {}),
    ("sum up clicks to category level", p.group_by, {"group_by": ["date", "cat_id"]}),
    ("filter out data from early 2018", p.filter_column, {"column": "date", "value": "2018-12-01"}),
    ("print data", p.print_df, {}),
    ("drop uninteresting colums", p.drop_columns, {"columns": ["internal_doc_id", "id_x", "manufacturer_id", "_version_", "internal_doc_id", "id_y", "adult"]}),
    ("print data", p.print_df, {}),
  ])
  logging.info('Load and process data pipeline')
  logging.info(load_and_process_data_pipeline.__str__())
  processed_data =load_and_process_data_pipeline.run()
  
  print(processed_data)
  logging.info('Finished')

if __name__ == '__main__':
  main()