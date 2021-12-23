import logging
import os
from typing import List

from matplotlib.figure import Figure

from src.data_types.i_model import IModel
from src.save_experiment_source.i_save_experiment_source import ISaveExperimentSource


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

    def save_metrics(self, metrics: List) -> None:
        with open(self.save_location + "/metrics.txt", "w") as f:
            f.writelines("MAE: 1.0 \nMSE:20.0")

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


def _combine_subfigure_titles(figure: Figure) -> str:
    titles = list(map(lambda ax: ax.title.get_text(), figure.axes))
    return ", ".join(titles)
