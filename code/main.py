from __future__ import annotations
import click
import logging
import pandas as pd
from utils import logger
from typing import Iterable
from prefect import task, Flow
from utils.config_parser import config, get_absolute_path
from pipelines import market_insight_pipelines as pipelines
from utils import neptune
from experiment import Experiment

@click.command()
@click.option('--experiment_description', '-e', default='', required=False, help='Experiment description.')
def main(experiment_description):
  config['experiment_description'] = experiment_description
  logger.init_logging()

  logging.info('Started')

  experiment = Experiment(experiment_description)
  experiment.choose_model_structure(config['model'].get())

  experiment.load_and_process_data(pipelines.market_insight_pipeline())

  experiment.train_model()
  experiment.test_model()
  experiment.save_model()

  # neptune_run = neptune.init_neptune()
  
  # params = {"learning_rate": 0.001, "optimizer": "Adam"}
  # neptune_run["parameters"] = params

  # for epoch in range(10):
  #     neptune_run["train/loss"].log(0.9 ** epoch)

  # neptune_run["eval/f1_score"] = 0.66

  # neptune_run.stop()


  logging.info('Finished')




def run_build_and_load_data_pipeline():
  pipeline = pipelines.market_insight_pipeline()
  logging.info(pipeline.__str__())
  pipeline.run() 

if __name__ == '__main__':
  main()