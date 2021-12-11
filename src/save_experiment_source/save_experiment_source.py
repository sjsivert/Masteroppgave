import logging
import os
from typing import List


class SaveExperimentSource:
    def __init__(self) -> None:
        pass

    def save_options(self, options: str) -> None:
        pass

    def save_models(self, models: List) -> None:
        pass

    def save_metrics(self, metrics: List) -> None:
        pass
