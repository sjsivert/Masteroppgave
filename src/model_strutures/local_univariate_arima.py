from typing import Dict

import pandas as pd
from genpipes.compose import Pipeline
from pandas.core.frame import DataFrame

from model_strutures.model_type import ModelType


class LocalUnivariateArima(ModelType):
    def __init__(self, model_options: Dict) -> None:
        NotImplementedError()

    def process_data(self, data_pipeline: Pipeline) -> DataFrame:
        pass

    def train(self) -> ModelType:
        pass

    def test(self) -> Dict:
        pass
