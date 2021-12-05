from __future__ import annotations
from typing import Dict, List
from genpipes.compose import Pipeline
import pandas as pd

from pandas import DataFrame

class ModelType():
  """
  Interface for all model types.
  Contains methods shared by all.
  """
  def __init__(self, model_options: Dict):
    pass

  def process_data(self, data_pipeline: Pipeline) -> DataFrame:
    """
    Processes data to get it on the correct format for the relevant model.
    args:
      data_pipeline: Pipeline object containing the data to be processed.
    """
    pass

  def train_model(self) -> ModelType:
    """
    Trains the model.
    """
    pass

  def test_model(self) -> Dict:
    """
    Tests the model.
    """
    pass
