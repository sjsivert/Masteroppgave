import logging
from pandas import DataFrame
from typing import Dict
from genpipes.compose import Pipeline
from model_strutures.model_type import ModelType
from model_strutures.local_univariate_arima import LocalUnivariateArima
from data_types.model_type_enum import ModelTypeEnum



class Experiment():
  """
  The main class for running experiments.
  It contains logic for choosing model type, logging results, and saving results.
  """
  def __init__(self, title: str, description: str) -> None:
    self.title = title
    self.experiment_description = description

  def choose_model_structure(self, model_options: Dict) -> ModelType:
    try:
      model_type = ModelTypeEnum[model_options['model_type']]
      if (model_type == ModelTypeEnum.local_univariate_arima):
        self.model = LocalUnivariateArima(model_options=model_options['arima'])
      return self.model

    except Exception as e:
      logging.error(f'Not a valid ModelType error: {e} \n \
        Valid ModelTypes are: {ModelTypeEnum.__members__}')
      raise e

  def load_and_process_data(self, data_pipeline: Pipeline) -> DataFrame:
    logging.info("Loading data")
    logging.info(data_pipeline.__str__())
    return self.model.process_data(data_pipeline)

  def train_model(self) -> ModelType:
    logging.info("Training model")
    return self.model.train_model()

  def test_model(self) -> Dict:
    logging.info("Testing model")
    return self.model.test_model()

  def save_model(self) -> None:
    logging.info("Saving model")
    pass
  

    