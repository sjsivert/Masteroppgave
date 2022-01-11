import logging
import os
import shutil
from typing import ContextManager, Dict, List

import neptune.new as neptune
from matplotlib.figure import Figure
from neptune.new.types import File
from src.save_experiment_source.i_save_experiment_source import ISaveExperimentSource
from src.save_experiment_source.save_local_disk_source import _combine_subfigure_titles
from src.utils.combine_subfigure_titles import _combine_subfigure_titles
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
                self.run[f"models/model_{idx}"].upload(File(f"temp_models/model_{idx}.pkl"), True)

    def save_metrics(self, metrics: Dict[str, Dict[str, float]]) -> None:
        average = {}
        for _, val in metrics.items():
            for error_name, error_value in val.items():
                if error_name not in average:
                    average[error_name] = []
                average[error_name].append(error_value)
        metrics["average"] = {i: sum(j) / len(j) for i, j in average.items()}
        self.run["metrics"] = metrics

    def save_figures(self, figures: List[Figure]):
        for figure in figures:
            title = _combine_subfigure_titles(figure)
            self.run[f"figures/fig_{title}"].upload(figure, True)

    def close(self) -> None:
        self.run.stop()
