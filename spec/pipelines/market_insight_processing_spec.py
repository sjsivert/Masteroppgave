import os
from pathlib import Path
from sys import path

from expects import be_true, equal, expect
from mamba import context, description, it
from pandas.core.frame import DataFrame
from src.pipelines.market_insight_pipelines import market_insight_pipeline
from src.pipelines.market_insight_processing import drop_columns
from src.utils.config_parser import get_absolute_path
