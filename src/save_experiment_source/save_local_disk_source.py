import logging
import os
from typing import Dict, List

from matplotlib.figure import Figure

from src.data_types.i_model import IModel
from src.save_experiment_source.i_save_experiment_source import ISaveExperimentSource
from src.utils.combine_subfigure_titles import _combine_subfigure_titles


class SaveLocalDiskSource(ISaveExperimentSource):
    def __init__(self, model_save_location: str, title) -> None:
        super().__init__()

        self.save_location = model_save_location + title
        try:
            logging.info(f"Saving models to {self.save_location}")
            os.mkdir(self.save_location)
        except FileExistsError:
            logging.warning(f"{self.save_location} already exists")
            raise FileExistsError

    def save_options(self, options: str) -> None:
        with open(self.save_location + "/options.yaml", "w") as f:
            f.write(options)

    def save_metrics(self, metrics: Dict[str, Dict[str, float]]) -> None:
        average = {}
        with open(self.save_location + "/metrics.txt", "w") as f:
            for test_name, test_value in metrics.items():
                f.writelines("\n_____Results dataset-{}_____\n".format(test_name))
                for metric_name in test_value.keys():
                    if metric_name not in average:
                        average[metric_name] = []
                    average[metric_name].append(test_value[metric_name])
                    f.writelines("{}: {}\n".format(metric_name, round(test_value[metric_name], 3)))

            f.writelines("\n___Average___\n")
            for metric_name, metric_value in average.items():
                f.writelines(
                    "{}: {}\n".format(metric_name, round(sum(metric_value) / len(metric_value), 3))
                )

    def save_models(self, models: List[IModel]) -> None:
        for idx, model in enumerate(models):
            model.save(self.save_location + f"/model_{idx}.pkl")

    def save_figures(self, figures: List[Figure]) -> None:
        for idx, figure in enumerate(figures):
            try:
                os.mkdir(self.save_location + f"/figures/")
            except FileExistsError:
                pass
            title = _combine_subfigure_titles(figure)
            figure.savefig(self.save_location + f"/figures/{title}.png")
