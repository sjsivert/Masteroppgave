import hashlib
import logging
import os
import shutil
from io import BytesIO
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
        load_from_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        self.save_location = Path(model_save_location).joinpath(title)
        self.checkpoint_save_location = checkpoint_save_location
        self.log_model_every_n_epoch = log_model_every_n_epoch

        if log_model_every_n_epoch > 0:
            self._wipe_and_init_checkpoint_save_location(title=title, description=description)
            self._save_options(options=options_dump, save_path=self.checkpoint_save_location)

        if not load_from_checkpoint:
            self._create_save_location()
            self._save_title_and_description(title=title, description=description)

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
        datasets: Dict[str, str],
        models: List[IModel],
        figures: List[Figure],
        data_pipeline_steps: str,
        experiment_tags: List[str],
    ) -> None:
        self._save_options(options)
        self._save_metrics(metrics)
        self._save_dataset_version(datasets)
        self._save_models(models)
        self._save_figures(figures)
        self._save_data_pipeline_steps(data_pipeline_steps)
        self._save_experiment_tags(experiment_tags)

    def load_model_and_metadata(self) -> None:
        # TODO: Method for fetching all data required for loading an experiment with models
        # TODO: Update parameters and return values
        pass

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

    def _save_data_pipeline_steps(self, data_pipeline_steps: str) -> None:
        with open(f"{self.save_location}/data_processing_steps.txt", "w") as f:
            f.write(data_pipeline_steps)

    def _save_dataset_version(self, datasets: Dict[str, str]) -> None:
        dataset_info = {}
        for file_type, file_path in datasets.items():
            path = Path(file_path)
            dataset_info[file_type] = {
                "name": path.name,
                "file_hash": SaveLocalDiskSource.generate_file_hash(path),
            }
        with open(f"{self.save_location}/datasets.json", "w") as f:
            f.write(dataset_info.__str__())

    @staticmethod
    def generate_file_hash(path: Path) -> str:
        hash_sha1 = hashlib.sha1()
        # Split into chunks to combat high use of memory
        chunk_size = 4096
        with open(path, "rb") as f:
            chunk = f.read(chunk_size)
            while len(chunk) > 0:
                hash_sha1.update(chunk)
                chunk = f.read(chunk_size)
        return hash_sha1.hexdigest()

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

    def _save_experiment_tags(self, tags: List[str]) -> None:
        with open(f"{self.save_location}/tags.txt", "a") as f:
            for tag in tags:
                f.write(f"{tag}\n")

    # Loading methods
    def _verify_dataset_version(self, datasets: Dict[str, str]) -> bool:
        # TODO: Check the hash of the given data path, and assert the same dataset is used
        pass

    def _fetch_dataset_version(self) -> str:
        # TODO: Fetch the hash stored
        pass

    def _load_models(self, models_path: List[Path]) -> List[BytesIO]:
        # TODO: Load the byte arrays of saved models
        pass

    def _load_config(self) -> Dict:
        # TODO: Load the old config, returning it as a dict, or whatever type is needed
        pass

    def _load_options(self) -> str:
        # TODO: Load options from save source
        pass

    ##########################################################
    # ILogTrainingSource interface
    ##########################################################
    def log_metrics(self, metrics: Dict[str, Dict[str, float]]) -> None:
        # TODO
        pass

    def log_models(self, models: List) -> None:
        # TODO
        pass

    def load_temp_models(self, models_path: List) -> None:
        # TODO
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

    def _save_title_and_description(self, title, description) -> None:
        with open(f"{self.save_location}/title-description.txt", "w") as f:
            f.write(f"{title}\n{description}")
