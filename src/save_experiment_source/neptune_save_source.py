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

    def load_model_and_metadata(self) -> None:
        # TODO: Method for fetching all data required for loading an experiment with models
        # TODO: Update parameters and return values
        pass

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
            for idx, model in enumerate(models):
                model.save("temp_models" + f"/model_{idx}.pkl")
                self.run[f"models/model_{idx}"].upload(File(f"temp_models/model_{idx}.pkl"), True)

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
        loaded_dataset_info = self._fetch_dataset_version()
        for file_type, file_path in datasets.items():
            path = Path(file_path)
            if file_type not in loaded_dataset_info.keys():
                return False
            elif loaded_dataset_info[file_type]["name"] != path.name:
                return False
            elif loaded_dataset_info[file_type][
                "file_hash"
            ] != SaveLocalDiskSource.generate_file_hash(path):
                return False
        return True

    def _fetch_dataset_version(self) -> str:
        # TODO: Fetch the hash stored
        pass

    def _load_models(self, models_path: List[Path]) -> List[BytesIO]:
        # TODO: Load the byte arrays of saved models
        pass

    def _load_options(self) -> str:
        # TODO: Load options from save source
        pass

    def _verify_pipeline_steps(self, data_pipeline_steps: str) -> bool:
        # TODO: Verify the pipeline steps are the same
        pass

    def _load_pipeline_steps(self) -> str:
        # TODO: Load pipeline steps from neptune save source
        pass

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
