from config_parser import config
import click
from prefect import task, Flow

@task
def print_experiment_description(experiment_name):
  print("experiment_name: {}".format(experiment_name))

@task
def print_data_loaction():
  print("hei")
  print(config['data']['data_path'].get())

def main_flow(experiment_description):
  with Flow("main") as flow:
    print_experiment_description(experiment_description)
    print_data_loaction()
    return flow

@click.command()
@click.option('--experiment_description', '-e', default='', required=False, help='Experiment description.')
def main(experiment_description):
  flow = main_flow(experiment_description)
  flow.run()

if __name__ == '__main__':
  main()