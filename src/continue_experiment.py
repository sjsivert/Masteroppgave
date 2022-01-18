import logging
from pathlib import Path
from typing import Optional

from src.experiment import Experiment
from src.utils.config_parser import config


class ContinueExperiment(Experiment):
    def __init__(
        self,
        experiment_checkpoints_location: Optional[Path] = Path(
            "./models/0_current_model_checkpoints/"
        ),
    ):
        self.experiment_checkpoints_location = experiment_checkpoints_location
        super().__init__()

    def continue_experiment(self) -> None:
        # Read experiment title and description from file
        with open(f"{self.experiment_checkpoints_location}/title-description.txt", "r") as f:
            self.title = f.readline().rstrip("\n")
            self.description = f.readline().rstrip("\n")

        logging.info(f"\nExperiment title: {self.title}\nDescription: {self.description}")

        # Clear and load old config
        config.clear()
        config.set_file(f"{self.experiment_checkpoints_location}/options.yaml")

        self._choose_model_structure(model_options=config["model"].get())

        # TODO: Find out how to load preprocessed data
        # model_structure.load_models(experiment_checkpoint_path)
        self._train_model()
        self._test_model()

    # TODO: Find out how to load data that has already been processed
    # cls._load_and_process_data(data_pipeline=data_pipeline)
