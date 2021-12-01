import click
import sys
import logging
from prefect import task, Flow
from config_parser import config, get_absolute_path
import data.data_loader as data_loader


def pipeline():
  with Flow("Main Pipeline") as flow:
    file_name = get_absolute_path(config['data']['data_path'].get())
    data = data_loader.load_data(file_name)
    return flow

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

  pipeline_flow = pipeline()
  pipeline_flow.run()

  logging.info('Finished')

if __name__ == '__main__':
  main()