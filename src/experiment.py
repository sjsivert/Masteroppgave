import logging
from pathlib import Path
from typing import Dict, List, Optional

from confuse.core import Configuration
from genpipes.compose import Pipeline
from pandas import DataFrame

from src.data_types.model_type_enum import ModelStructureEnum
from src.model_strutures.i_model_structure import IModelStructure
from src.model_strutures.local_univariate_arima_structure import LocalUnivariateArimaStructure
from src.model_strutures.validation_model_structure import ValidationModelStructure
from src.save_experiment_source.i_save_experiment_source import ISaveExperimentSource
from src.save_experiment_source.save_local_disk_source import SaveLocalDiskSource
from src.utils.config_parser import config


class Experiment:
    """
    The main class for running experiments.
    It contains logic for choosing model type, logging results, and saving results.
    """

    def __init__(
        self,
        title: str,
        description: str,
        save_sources_to_use=[],
        save_source_options={},
    ) -> None:
        self.model_structure = None
        self.title = title
        self.experiment_description = description
        self.save_sources = self._init_save_sources(save_sources_to_use, save_source_options)

    def _init_save_sources(
        self, save_sources: List[str], save_source_options: Dict
    ) -> List[ISaveExperimentSource]:
        sources = []
        for source in save_sources:
            if source == "disk":
                sources.append(
                    SaveLocalDiskSource(
                        **save_source_options["disk"],
                        options_dump=config.dump(),
                        title=self.title,
                        description=self.experiment_description,
                    )
                )
            # TODO: Add Neptune save source
        return sources

    def run_complete_experiment(
        self,
        model_options: Dict,
        data_pipeline: Pipeline,
        save: bool = True,
        options_to_save: Optional[str] = None,
    ) -> None:
        """
        Run a complete experiment with preprocessing of data, training,testing and optional saving.
        """
        logging.info(f"Running complete experiment with saving set to {save}")
        logging.info(data_pipeline.__str__())

        self._choose_model_structure(model_options=model_options)

        self._load_and_process_data(data_pipeline=data_pipeline)

        self._train_model()
        self._test_model()

        if save and options_to_save:
            self._save_model(options=options_to_save)

    @classmethod
    def continue_experiment(cls, experiment_checkpoint_path: Path) -> None:
        # Read experiment title and description from file
        title = ""
        description = ""
        with open(f"{experiment_checkpoint_path}/title-description.txt", "r") as f:
            title = f.readline().rstrip("\n")
            description = f.readline().rstrip("\n")

        # Clear and load old config
        config.clear()
        config.set_file(f"{experiment_checkpoint_path}/options.yaml")

        # model_structure = cls._choose_model_structure(model_options=config["model"].get())

        # TODO: Find out how to load preprocessed data
        # model_structure.load_models(experiment_checkpoint_path)
        cls._train_model()
        cls._test_model()

    # TODO: Find out how to load data that has already been processed
    # cls._load_and_process_data(data_pipeline=data_pipeline)

        logging.warning("Not implemented")
        # TODO: Implement
        pass
    def _choose_model_structure(self, model_options: Dict) -> IModelStructure:
        try:
            model_structure = ModelStructureEnum[model_options["model_type"]]
            if model_structure == ModelStructureEnum.validation_model:
                self.model_structure = ValidationModelStructure(self.save_sources)
            elif model_structure == ModelStructureEnum.local_univariate_arima:
                self.model_structure = LocalUnivariateArimaStructure(
                    self.save_sources, **model_options["local_univariate_arima"]
                )
            return self.model_structure

        except Exception as e:
            logging.error(
                f"Not a valid ModelType error: {e} \n \
                Valid ModelTypes are: {ModelStructureEnum.__members__}"
            )
            raise e

    def _load_and_process_data(self, data_pipeline: Pipeline) -> DataFrame:
        logging.info("Loading data")
        logging.info(data_pipeline.__str__())
        return self.model_structure.process_data(data_pipeline)

    def _train_model(self) -> IModelStructure:
        logging.info("Training model")
        return self.model_structure.train()

    def _test_model(self) -> Dict:
        logging.info("Testing model")
        return self.model_structure.test()

    def _save_model(self, options: str) -> None:
        """
        Save the model and correspoding experiment to an already existing directory.
        """
        logging.info("Saving model")
        for save_source in self.save_sources:
            save_source.save_options(options)

            # TODO: Save model

            # TODO: Get metrics from model and pass to save_source
            save_source.save_metrics({})

            # TODO: Save hyperparameters

            # TODO: Save figures

            # Save predictions
            # TODO: Implement
