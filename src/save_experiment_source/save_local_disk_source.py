import logging
import os
from typing import Dict, List

from src.save_experiment_source.save_experiment_source import \
    SaveExperimentSource


class SaveLocalDiskSource(SaveExperimentSource):
    def __init__(self, options: Dict, title) -> None:
        self.save_location = options['model_save_location'] + title
        try:
            logging.info(f'Saving models to {self.save_location}')
            os.mkdir(self.save_location)
        except FileExistsError:
            logging.warn(f"{self.save_location} already exists")
            raise FileExistsError

    def save_options(self, options: str) -> None:
        with open(self.save_location + "/options.yaml", "w") as f:
            f.write(options)

    def save_metrics(self, metrics: List) -> None:
        with open(self.save_location + "/metrics.txt", "w") as f:
            f.writelines("MAE: 1.0 \nMSE:20.0")

    def save_models(self, models: List) -> None:
        # TODO: Implement
        pass

