import json
from plotly.graph_objs import Figure as PlotlyFigure

import logging
import os
import shutil
from abc import ABC
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from matplotlib.figure import Figure
from pandas import DataFrame
from src.data_types.i_model import IModel
from src.save_experiment_source.i_save_experiment_source import ISaveExperimentSource
from src.utils.combine_subfigure_titles import combine_subfigure_titles
from src.utils.config_parser import config
from src.utils.file_hasher import generate_file_hash


class SaveLocalDiskSource(ISaveExperimentSource, ABC):
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
        data_pipeline_steps: List[str],
        experiment_tags: List[str],
        tuning: Dict,
        predictions: Optional[DataFrame] = None,
    ) -> None:
        self._save_options(options)
        self._save_metrics(metrics)
        self._save_dataset_version(datasets)
        self._save_models(models)
        self._save_figures(figures)
        self._save_data_pipeline_steps(data_pipeline_steps)
        self._save_experiment_tags(experiment_tags)
        self._save_tuning_metrics(tuning)
        self._save_log()
        self._save_predictions(predictions)

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
            for model_name, model_metrics in metrics.items():
                f.writelines("\n_____Results dataset-{}_____\n".format(model_name))
                for metric_name in model_metrics.keys():
                    f.writelines(f"{metric_name}: {round(model_metrics[metric_name], 5)}\n")

    def _save_data_pipeline_steps(self, data_pipeline_steps: List[str]) -> None:
        with open(f"{self.save_location}/data_processing_steps.txt", "w") as f:
            for pipeline in data_pipeline_steps:
                f.write(pipeline)

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

    def _save_models(self, models: List[IModel], custom_save_path: Path = None) -> None:
        save_path = custom_save_path if custom_save_path else self.save_location
        for model in models:
            model.save(path=f"{save_path}/")

    def _save_figures(self, figures: List[Figure]) -> None:
        for figure in figures:
            try:
                os.mkdir(f"{self.save_location}/figures/")
            except FileExistsError:
                pass
            title = combine_subfigure_titles(figure)

            if type(figure) is PlotlyFigure:
                figure.write_image(f"{self.save_location}/figures/{title}.png")
            else:
                figure.savefig(f"{self.save_location}/figures/{title}.png")

    def _save_experiment_tags(self, tags: List[str]) -> None:
        with open(f"{self.save_location}/tags.txt", "a") as f:
            for tag in tags:
                f.write(f"{tag}\n")

    def _save_tuning_metrics(self, tuning: Dict) -> None:
        if tuning is None or not tuning:
            return
        with open(f"{self.save_location}/tuning.txt", "w") as f:
            f.write("Parameter Tuning results")
            for model_name, param_error_set in tuning.items():
                f.write(f"\n\n\nModel tuning: Dataset {model_name}")
                for params, err in param_error_set.items():
                    error_values = ",".join([f"{x}:{y}" for x, y in err.items()])
                    f.write(f"\n\nParameters: {params}\n Error: {error_values}")

    def _save_log(self) -> None:
        # TODO Make log file configurable
        try:
            shutil.copyfile(
                f"{config['logger']['log_file'].get()}", f"{self.save_location}/log_file.log"
            )
        except FileNotFoundError:
            pass

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

    def load_model_and_metadata(self) -> None:
        raise NotImplementedError()

    def _save_title_and_description(self, title, description) -> None:
        with open(f"{self.save_location}/title-description.txt", "w") as f:
            f.write(f"{title}\n{description}")

    def _save_predictions(self, predictions: DataFrame) -> None:
        if predictions is not None:
            predictions.to_csv(f"{self.save_location}/predictions.csv")
