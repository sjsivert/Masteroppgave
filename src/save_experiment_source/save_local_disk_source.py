import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from matplotlib.figure import Figure

from src.data_types.i_model import IModel
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from src.save_experiment_source.i_save_experiment_source import ISaveExperimentSource
from src.utils.combine_subfigure_titles import combine_subfigure_titles


class SaveLocalDiskSource(ISaveExperimentSource, ILogTrainingSource):
    def __init__(
        self,
        model_save_location: Path,
        title: str,
        description: str = "",
        options_dump: str = "",
        checkpoint_save_location: Path = Path("models/0_current_model_checkpoints/"),
        log_model_every_n_epoch: int = 0,
        tags: List[str] = [],
    ) -> None:
        super().__init__()

        self.save_location = Path(model_save_location).joinpath(title)
        self.checkpoint_save_location = checkpoint_save_location
        self.log_model_every_n_epoch = log_model_every_n_epoch

        self._create_save_location()
        self.save_experiment_tags(tags)

        if log_model_every_n_epoch > 0:
            self._wipe_and_init_checkpoint_save_location(title=title, description=description)
            self._save_options(options=options_dump, save_path=self.checkpoint_save_location)

    def _create_save_location(self):
        try:
            logging.info(f"Creating model save location {self.save_location}")
            os.mkdir(self.save_location.__str__())
        except FileExistsError:
            logging.warning(f"{self.save_location} already exists")
            raise FileExistsError

    def save_model_and_metadata(
        self,
        options: str,
        metrics: Dict[str, Dict[str, float]],
        models: List[IModel],
        figures: List[Figure],
    ) -> None:
        self._save_options(options)
        self._save_metrics(metrics)
        self._save_models(models)
        self._save_figures(figures)

    def _save_options(self, options: str, save_path: Optional[Path] = None) -> None:
        """
        Saves the options used to train the model.
        If save_path is not provided saves to the pre-defined save_location.
        :param options:
        :param save_path:
        :return:
        """
        path = save_path if save_path else self.save_location
        with open(f"{path}/options.yaml", "w") as f:
            f.write(options)

    def _save_metrics(self, metrics: Dict[str, Dict[str, float]]) -> None:
        average = {}
        with open(f"{self.save_location}/metrics.txt", "w") as f:
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

    def _save_models(self, models: List[IModel]) -> None:
        for idx, model in enumerate(models):
            model.save(f"{self.save_location}/model_{idx}.pkl")

    def _save_figures(self, figures: List[Figure]) -> None:
        for idx, figure in enumerate(figures):
            try:
                os.mkdir(f"{self.save_location}/figures/")
            except FileExistsError:
                pass
            title = combine_subfigure_titles(figure)
            figure.savefig(f"{self.save_location}/figures/{title}.png")

    def save_experiment_tags(self, tags: List[str]) -> None:
        with open(f"{self.save_location}/tags.txt", "a") as f:
            for tag in tags:
                f.write(f"{tag}\n")

    # ILogTrainingSource interface
    def log_metrics(self, metrics: Dict[str, Dict[str, float]]) -> None:
        # Interface, not to be implemented
        pass

    def log_models(self, models: List) -> None:
        # Interface, not to be implemented
        pass

    def load_temp_models(self, models_path: List) -> None:
        # Interface, not to be implemented
        return None

    def _wipe_and_init_checkpoint_save_location(self, title: str, description: str) -> None:
        logging.info(
            f"Wiping and initializing checkpoint save location {self.checkpoint_save_location}"
        )
        try:
            shutil.rmtree(self.checkpoint_save_location)
        except FileNotFoundError:
            pass

        os.mkdir(self.checkpoint_save_location)

        # Save the title and description to temp location
        with open(f"{self.checkpoint_save_location}/title-description.txt", "w") as f:
            f.write(f"{title}\n{description}")
