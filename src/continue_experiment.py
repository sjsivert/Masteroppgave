import logging
from pathlib import Path
from typing import Optional

from src.experiment import Experiment
from src.utils.config_parser import config


class ContinueExperiment(Experiment):
    """
    Class for continuing an already initialized experiment.
    It assumes that the previous experiment has been initialized and saved
    to the checkpoints_save_location specified in the config.
    """

    def __init__(
        self,
        experiment_checkpoints_location: Optional[Path] = Path(
            "./models/0_current_model_checkpoints/"
        ),
    ):
        assert experiment_checkpoints_location.is_dir(), FileNotFoundError(
            f"experiment_checkpoints_location does not exist: {experiment_checkpoints_location}"
        )

        self.neptune_id_to_load = None
        self.experiment_checkpoints_location = experiment_checkpoints_location
        super().__init__()

    def _load_neptune_id_from_checkpoint_location(self) -> str:
        with open(f"{self.experiment_checkpoints_location}/neptune_id.txt", "r") as f:
            neptune_id = f.readline().rstrip("\n")
            return neptune_id

    def continue_experiment(self) -> None:
        self.title, self.description = self._load_title_and_description()
        logging.info(f"\nExperiment title: {self.title}\nDescription: {self.description}")

        self._load_saved_options()

        neptune_save_source_was_used = (
            "neptune" in config["experiment"]["save_sources_to_use"].get()
        )
        if neptune_save_source_was_used:
            self.neptune_id_to_load = self._load_neptune_id_from_checkpoint_location()
            logging.info(f"Neptune experiment id: {self.neptune_id_to_load}")

        self._choose_model_structure(model_options=config["model"].get())

        save_sources_to_use = config["experiment"]["save_sources_to_use"].get()
        save_source_options = config["experiment"]["save_source"].get()

        self._init_save_sources(
            save_sources_to_use=save_sources_to_use,
            save_source_options=save_source_options,
            load_from_checkpoint=True,
            neptune_id_to_load=self.neptune_id_to_load,
        )

        """
        ___ Loading model structures ___
        In order to load previous model structures from prior experiments, we call the _load_models
        method from the selected save source. This methods takes a list of models as input.
        This list of models is derived from the created models structure using the 'get_models' method.
        Each instanciated model contained in the list should have the same unique model name 'model.name'
        as the model that was saved had. This is in order to load the correct model.
        """

        # TODO!
        # model_structure.load_models(experiment_checkpoint_path)
        self._train_model()
        self._test_model()

    def continue_tuning(self) -> None:
        # Load prev data from ran exp
        self.title, self.description = self._load_title_and_description()
        logging.info(f"\nExperiment title: {self.title}\nDescription: {self.description}")
        self._load_saved_options()
        neptune_save_source_was_used = (
            "neptune" in config["experiment"]["save_sources_to_use"].get()
        )
        if neptune_save_source_was_used:
            self.neptune_id_to_load = self._load_neptune_id_from_checkpoint_location()
            logging.info(f"Neptune experiment id: {self.neptune_id_to_load}")

        self._choose_model_structure(model_options=config["model"].get())

        save_sources_to_use = config["experiment"]["save_sources_to_use"].get()
        save_source_options = config["experiment"]["save_source"].get()

        self._init_save_sources(
            save_sources_to_use=save_sources_to_use,
            save_source_options=save_source_options,
            load_from_checkpoint=True,
            neptune_id_to_load=self.neptune_id_to_load,
        )

        """
        ___ Loading tuning info ___
        Load data of tuned models and what remains.
        """
        loaded_tuning_param_error_sets = self._load_tuning_info()
        self.model_structure.tuning_parameter_error_sets = loaded_tuning_param_error_sets
        # Continue tuning
        self.model_structure.auto_tuning()

        # TODO! Saving tuned model
        # if save and options_to_save:
        #     self._save_model(options=options_to_save)

    def _load_title_and_description(self) -> (str, str):
        try:
            with open(f"{self.experiment_checkpoints_location}/title-description.txt", "r") as f:
                title = f.readline().rstrip("\n")
                description = f.readline().rstrip("\n")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not load title and description from: {self.experiment_checkpoints_location}"
            )

        return title, description

    def _load_saved_options(self) -> None:
        config.clear()
        config.set_file(f"{self.experiment_checkpoints_location}/options.yaml")

    def _load_tuning_info(self):
        tuning_info = None
        for save_source in self.save_sources:
            tuning_info = save_source.load_tuning_metrics()
            if tuning_info is not None:
                break
        return tuning_info if tuning_info is not None else {}
