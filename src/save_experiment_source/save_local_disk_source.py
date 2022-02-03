import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from matplotlib.figure import Figure
from src.data_types.i_model import IModel
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from src.save_experiment_source.i_save_experiment_source import ISaveExperimentSource
from src.utils.combine_subfigure_titles import combine_subfigure_titles
from src.utils.file_hasher import generate_file_hash


class SaveLocalDiskSource(ISaveExperimentSource, ILogTrainingSource):
    def __init__(
        self,
        model_save_location: Path,
        title: str,
        description: str = "",
        options_dump: str = "",
        load_from_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        self.save_location = Path(model_save_location).joinpath(title)

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

    def load_metadata(
        self, datasets: Dict[str, Dict[str, float]], data_pipeline_steps: str
    ) -> Tuple[str, bool, bool]:
        """
        :return: (str: Stored options, bool: Dataset version validation, bool: Pipeline step validation)
        """
        return (
            self._load_options(),
            self._verify_dataset_version(datasets),
            self._verify_pipeline_steps(data_pipeline_steps),
        )

    def _save_options(self, options: str, save_path: Optional[Path] = None) -> None:
        """
        Saves the options used to train the model.
        If save_path is not provided saves to the pre-defined save_location.
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
                "file_hash": generate_file_hash(path),
            }
        with open(f"{self.save_location}/datasets.json", "w") as f:
            json.dump(dataset_info, f)

    def _save_models(self, models: List[IModel]) -> None:
        for model in models:
            model.save(path=f"{self.save_location}/")

    def _save_figures(self, figures: List[Figure]) -> None:
        for figure in figures:
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
        """
        Verify data and file name is the same
        """
        loaded_dataset_info = self._fetch_dataset_version()
        for file_type, file_path in datasets.items():
            path = Path(file_path)
            if (
                file_type not in loaded_dataset_info.keys()
                or loaded_dataset_info[file_type]["name"] != path.nam
                or loaded_dataset_info[file_type]["file_hash"] != generate_file_hash(path)
            ):
                return False
        return True

    def _fetch_dataset_version(self) -> str:
        if not os.path.exists(f"{self.save_location}/datasets.json"):
            raise FileNotFoundError(
                f"{self.save_location}/datasets.json, is not found in the model store."
            )
        with open(f"{self.save_location}/datasets.json", "r") as f:
            loaded_dataset_info = json.load(f)
        return loaded_dataset_info

    def _load_models(self, models: List[IModel]) -> None:
        for idx, model in enumerate(models):
            model.load(path=f"{self.save_location}/")

    def _load_options(self, save_path: Optional[str] = None) -> str:
        path = save_path if save_path else self.save_location
        if not os.path.exists(f"{path}/options.yaml"):
            raise FileNotFoundError("Stored options file not found")

        with open(f"{path}/options.yaml", "r") as f:
            options_contents = f.read()
        return options_contents

    def _verify_pipeline_steps(self, data_pipeline_steps: str) -> bool:
        loaded_pipeline_steps = self._load_pipeline_steps()
        return loaded_pipeline_steps == data_pipeline_steps

    def _load_pipeline_steps(self) -> str:
        if not os.path.exists(f"{self.save_location}/data_processing_steps.txt"):
            raise FileNotFoundError(
                f"Could not find: {self.save_location}/data_processing_steps.txt"
            )
        with open(f"{self.save_location}/data_processing_steps.txt", "r") as f:
            pipeline_steps = f.read()
        return pipeline_steps

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

    def _save_title_and_description(self, title, description) -> None:
        with open(f"{self.save_location}/title-description.txt", "w") as f:
            f.write(f"{title}\n{description}")
