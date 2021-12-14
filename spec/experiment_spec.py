import os
import shutil
from types import ModuleType

import expects
import pytest
from confuse.exceptions import NotFoundError
from expects import be, be_false, be_true, expect
from genpipes.compose import Pipeline
from mamba import after, before, description, it
from mockito import mock, when
from pandas.core.frame import DataFrame
from src import main
from src.data_types.model_type_enum import ModelTypeEnum
from src.experiment import Experiment
from src.model_strutures.i_model_type import IModelType
from src.utils.config_parser import config, get_absolute_path
from src.utils.logger import init_logging

with description("Experiment") as self:
    with it("initialises with title and description"):
        experiment = Experiment("title", "description")
        expect(experiment.title).to(be("title"))
        expect(experiment.experiment_description).to(be("description"))

    with it("returns dataframe on load_and_process_data()"):
        experiment = Experiment("title", "description")
        pipeline = mock(Pipeline)
        model = mock(IModelType)
        df = DataFrame({"a": [1, 2, 3]})
        when(model).process_data(pipeline).thenReturn(df)
        experiment.model = model

        dataframe = experiment.load_and_process_data(pipeline)
        expect(dataframe).to(be(df))

    with it("can choose_model_structure arima"):
        experiment = Experiment("title", "description")
        options = {
            "model_type": "local_univariate_arima",
            "arima": {"order": (1, 1, 1)},
        }
        experiment.choose_model_structure(options)
        expect(experiment.model).to_not(expects.be_none)

