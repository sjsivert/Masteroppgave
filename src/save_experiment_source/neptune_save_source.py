import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import neptune.new as neptune
from matplotlib.figure import Figure
from neptune.new.types import File
from src.data_types.i_model import IModel
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from src.save_experiment_source.i_save_experiment_source import ISaveExperimentSource
from src.utils.combine_subfigure_titles import combine_subfigure_titles
from src.utils.file_hasher import generate_file_hash
from src.utils.temporary_files import temp_files


class NeptuneSaveSource(ISaveExperimentSource, ILogTrainingSource):
    """
    Neptune save source for tracking ML experiments.
    """

    def __init__(
        self,
        project_id: str,
        title,
        description,
        load_from_checkpoint: bool = False,
        load_run_id: str = None,
        **xargs,
    ) -> None:
        super().__init__()
        if not load_from_checkpoint:
            logging.info("Creating new Neptune experiment")
            self.run = neptune.init(project=project_id, custom_run_id=title, **xargs)
            self.run_url = self.run.get_run_url()
            self.run["sys/tags"].add(["Experiment"])

            self.run["sys/name"] = title
            self.run["Experiment title"] = title
            self.run["Experiment description"] = description
        elif load_from_checkpoint and load_run_id is not None:
            logging.info(f"Loaded preview Neptune Experiment run: {load_run_id} from checkpoint")
            self.run = neptune.init(project=project_id, run=load_run_id, **xargs)

        logging.info(f"Neptune run URL: {self.run.get_run_url()}")

    def get_run_id(self) -> str:
        return self.run_url.split("/")[-1]

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
    ) -> None:
        """
        :return: 3 variables,
        1 -> The saved options
        2 -> A boolean value signaling if the stored and current dataset is the same
        3 -> A boolean value signaling if the stored and current pipelines are the same
        """
        return (
            self._load_options(),
            self._verify_dataset_version(datasets),
            self._verify_pipeline_steps(data_pipeline_steps),
        )

    def _save_options(self, options: str) -> None:
        self.run["options"] = options

    def _save_metrics(self, metrics: Dict[str, Dict[str, float]]) -> None:
        average = {}
        for _, val in metrics.items():
            for error_name, error_value in val.items():
                if error_name not in average:
                    average[error_name] = []
                average[error_name].append(error_value)
        metrics["average"] = {i: sum(j) / len(j) for i, j in average.items()}
        self.run["metrics"] = metrics

    def _save_data_pipeline_steps(self, data_pipeline_steps: str) -> None:
        self.run["data_pipeline_steps"] = data_pipeline_steps

    def _save_dataset_version(self, datasets: Dict[str, str]) -> None:
        for data_type, data_path in datasets.items():
            path = Path(data_path)
            self.run[f"datasets/{data_type}_name"] = path.name
            self.run[f"datasets/{data_type}"].track_files(data_path, wait=True)

    def _save_models(self, models: List) -> None:
        with temp_files("temp_models"):
            for model in models:
                model.save("temp_models" + f"/model_{model.get_name()}.pkl")
                self.run[f"models/model_{model.get_name()}"].upload(
                    File(f"temp_models/model_{model.get_name()}.pkl"), True
                )

    def _save_figures(self, figures: List[Figure]):
        for figure in figures:
            title = combine_subfigure_titles(figure)
            self.run[f"figures/fig_{title}"].upload(figure, True)

    def _save_experiment_tags(self, tags: List[str]) -> None:
        for tag in tags:
            self.run["sys/tags"].add([tag])

    def close(self) -> None:
        self.run.stop()

    # Loading methods
    def _verify_dataset_version(self, datasets: Dict[str, str]) -> bool:
        """
        Verify data and file name is the same
        """
        for file_type, file_path in datasets.items():
            file_hash, file_name = self._fetch_dataset_version(file_type)
            path = Path(file_path)
            if file_hash == "" or file_name == "" or file_hash is None or file_name is None:
                return False
            elif file_name != path.name:
                return False
            elif file_hash != generate_file_hash(path):  # TODO: How to get same hash?
                return False
        return True

    def _fetch_dataset_version(self, data_type_name: str) -> (str, str):
        """
        :return: 2 values. First value is file hash, second is the file name
        """
        return (
            self.run[f"datasets/{data_type_name}"].fetch_hash(),
            self.run[f"datasets/{data_type_name}_name"],
        )

    def _load_models(self, models: List[IModel]) -> None:
        # Load models from neptune to temp folder, before loading data to models

        base_path = f"{self.save_location}/model_loading/"
        with temp_files(base_path):
            for model in models:
                self.run[f"models/model_{model.get_name()}"].download(base_path)
                model.load(path=f"{base_path}")

    def _load_options(self) -> str:
        # Load options from save source
        return self.run["options"].fetch()

    def _verify_pipeline_steps(self, data_pipeline_steps: str) -> bool:
        # Verify the pipeline steps are the same
        return self._load_pipeline_steps() == data_pipeline_steps

    def _load_pipeline_steps(self) -> str:
        # Load pipeline steps from neptune save source
        return self.run["data_pipeline_steps"].fetch()

    ##########################################################
    # ILogTrainingSource interface
    ##########################################################
    def log_metrics(self, metrics: Dict[str, Dict[str, float]]) -> None:
        # Interface, not to be implemented
        pass

    def log_models(self, models: List) -> None:
        # Interface, not to be implemented
        pass

    def load_temp_models(self, models_path: List) -> None:
        # Interface, not to be implemented
        return None
