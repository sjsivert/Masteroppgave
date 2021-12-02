import click
import logging
import pandas as pd
from utils import logger
from typing import Iterable
from prefect import task, Flow
from utils.config_parser import config, get_absolute_path
from pipelines import market_insight_pipelines as pipelines
from utils import neptune

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

  # TODO: Build model

  # TODO: Train model

  # TODO: Validata model
  neptune_run = neptune.init_neptune()
  
  # params = {"learning_rate": 0.001, "optimizer": "Adam"}
  # neptune_run["parameters"] = params

  # for epoch in range(10):
  #     neptune_run["train/loss"].log(0.9 ** epoch)

  # neptune_run["eval/f1_score"] = 0.66

  neptune_run.stop()


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