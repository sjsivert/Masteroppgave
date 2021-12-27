import logging
import os
import shutil
from typing import ContextManager, List

import neptune.new as neptune
from matplotlib.figure import Figure
from src.save_experiment_source.i_save_experiment_source import \
    ISaveExperimentSource
from src.save_experiment_source.save_local_disk_source import \
    _combine_subfigure_titles
from src.utils.temporary_files import temp_files


class NeptuneSaveSource(ISaveExperimentSource):
    """
    Neptune save source for tracking ML experiments.
    """

    def __init__(self, project_id: str, title, description, tags=[]) -> None:
        self.run = neptune.init(project=project_id)

        self.run["sys/tags"].add(["Experiment"])
        for tag in tags:
            self.run["sys/tags"].add([tag])

        self.run["sys/name"] = title
        self.run["Experiment title"] = title
        self.run["Experiment description"] = description
        logging.info(f"Starting logging neptune experiment: {self.run.get_run_url()}")

    def save_options(self, options: str) -> None:
        self.run["options"] = options

    def save_models(self, models: List) -> None:
        with temp_files("temp_models"):
            for idx, model in enumerate(models):
                model.save("temp_models" + f"/model_{idx}.pkl")
                self.run["model"].upload(f"temp_models/model_{idx}")

    def save_metrics(self, metrics: List) -> None:
        self.run["metrics"] = metrics

    def save_figures(self, figures: List[Figure]):
        # TODO: This might not work yet 
        # Because the files gets deleted before neptune uploads
        with temp_files("temp_figures"):
            for figure in figures:
                title = _combine_subfigure_titles(figure)
                figure.savefig(f"temp_figures/{title}.png")
                self.run["aux/figures"].upload(f"temp_figures/{title}.png")


    def close(self) -> None:
        self.run.stop()
