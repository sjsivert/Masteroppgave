import click
import logging
import pandas as pd
from utils import logger
from typing import Iterable
from prefect import task, Flow
from utils.config_parser import config, get_absolute_path
from pipelines import market_insight_pipelines as pipelines

@click.command()
@click.option('--experiment_description', '-e', default='', required=False, help='Experiment description.')
def main(experiment_description):
  config['experiment_description'] = experiment_description
  logger.init_logging()

  logging.info('Started')

  # Load and preprocess data
  pipeline = pipelines.build_load_and_process_data_pipeline()
  logging.info(pipeline.__str__())
  processed_data = pipeline.run() 

  split_data_to_train_and_test(processed_data)

  # TODO: Train model

  # TODO: Validata model
  
  print(processed_data)


  logging.info('Finished')



def split_data_to_train_and_test(data: pd.DataFrame):
  # TODO: Implement
  return (data, data)

def run_build_and_load_data_pipeline():
  pipeline = pipelines.build_load_and_process_data_pipeline()
  logging.info(pipeline.__str__())
  pipeline.run() 

if __name__ == '__main__':
  main()