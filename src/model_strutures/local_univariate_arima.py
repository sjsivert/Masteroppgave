from typing import Dict, Tuple

import pandas as pd
from genpipes.compose import Pipeline
from pandas.core.frame import DataFrame
from src.model_strutures.i_model_type import IModelType


class LocalUnivariateArima(IModelType):
    def __init__(self, order: Tuple[int]) -> None:
        # TODO: Implement
        self.order = order

    def process_data(self, data_pipeline: Pipeline) -> DataFrame:
        # TODO: Update with additional data processing
        return data_pipeline.run()

    def train(self) -> IModelType:
        # TODO: Implement
        raise NotImplementedError()

    def test(self) -> Dict:
        # TODO: Implement
        raise NotImplementedError()
