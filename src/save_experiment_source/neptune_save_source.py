import logging
import os
import shutil
from typing import Dict, List

import neptune.new as neptune
from src.save_experiment_source.i_save_experiment_source import \
    ISaveExperimentSource


class NeptuneSaveSource(ISaveExperimentSource):
    """
    Neptune save source for tracking ML experiments.
    """

    def __init__(self, project_id: str, title, description, tags: List[str] = []) -> None:
        self.run = neptune.init(project=project_id)

        self.run['sys/tags'].add(['Experiment'])
        for tag in tags:
            self.run['sys/tags'].add([tag])

        self.run['sys/name'] = title
        self.run['Experiment title'] = title
        self.run['Experiment description'] = description
        logging.info(f'Starting logging neptune experiment: {self.run.get_run_url()}')

    def save_options(self, options: str) -> None:
        self.run['options'] = options

    def save_models(self, models: List) -> None:
        # Make temporary folder for saving models
        os.mkdir('temp_models')
        for idx, model in enumerate(models):
            model.save('temp_models' + f"/model_{idx}.pkl")
            self.run['model'].upload(f'temp_models/model_{idx}')

        # Remove temporary folder and models
        shutil.rmtree('temp_models')

    def save_metrics(self, metrics: List) -> None:
        self.run['metrics'] = metrics
