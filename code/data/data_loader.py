import pandas as pd
import logging
from prefect import task
from main import EXPERIMENT_DESCRIPTION
from config_parser import config

@task
def load_data(filename: str) -> pd.DataFrame:
    """
    Loads data from a file.

    :param filename: name of the file to load data from
    :return: pandas dataframe containing the data
    """
    logging.info("Loading data from file: {}".format(filename))
    return pd.read_csv(filename)
