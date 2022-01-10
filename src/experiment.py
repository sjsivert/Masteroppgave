import logging
from typing import Dict, List

from confuse.core import Configuration
from genpipes.compose import Pipeline
from pandas import DataFrame

from src.data_types.model_type_enum import ModelTypeEnum
from src.model_strutures.i_model_type import IModelType
from src.model_strutures.local_univariate_arima import LocalUnivariateArima
from src.save_experiment_source.i_save_experiment_source import ISaveExperimentSource
from src.save_experiment_source.save_local_disk_source import SaveLocalDiskSource


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
        self.model = None
        self.title = title
        self.experiment_description = description
        self.save_sources = self._init_save_sources(save_sources_to_use, save_source_options)

    def _init_save_sources(
        self, save_sources: List[str], save_source_options: Dict
    ) -> List[ISaveExperimentSource]:
        sources = []
        for source in save_sources:
            if source == "disk":
                sources.append(SaveLocalDiskSource(**save_source_options["disk"], title=self.title))
        return sources

    def run_complete_experiment_with_saving(
        self, model_options: Dict, data_pipeline: Pipeline, options_to_save=Configuration
    ) -> None:
        """
        Run a complete experiment with preprocessing of data, training, testing, and saving.
        """
        logging.info("Running complete experiment with saving")
        logging.info(data_pipeline.__str__())

        self._choose_model_structure(model_options=model_options)

        self._load_and_process_data(data_pipeline=data_pipeline)

        self._train_model()
        self._test_model()
        self._save_model(options=options_to_save.dump())

    def _choose_model_structure(self, model_options: Dict) -> IModelType:
        try:
            model_type = ModelTypeEnum[model_options["model_type"]]
            if model_type == ModelTypeEnum.local_univariate_arima:
                self.model = LocalUnivariateArima(**model_options["local_univariate_arima"])
            return self.model

        except Exception as e:
            logging.error(
                f"Not a valid ModelType error: {e} \n \
                Valid ModelTypes are: {ModelTypeEnum.__members__}"
            )
            raise e

    def _load_and_process_data(self, data_pipeline: Pipeline) -> DataFrame:
        logging.info("Loading data")
        logging.info(data_pipeline.__str__())
        return self.model.process_data(data_pipeline)

    def _train_model(self) -> IModelType:
        logging.info("Training model")
        # TODO: Implement
        return self.model.train_model()

    def _test_model(self) -> Dict:
        logging.info("Testing model")
        # TODO: Implement
        return self.model.test_model()

    def _save_model(self, options: str) -> None:
        """
        Save the model and correspoding experiment to an already existing directory.
        """
        logging.info("Saving model")
        for save_source in self.save_sources:
            save_source.save_options(options)
            # Save options

            # TODO: Save models

            # Save metrics
            save_source.save_metrics([])

            # TODO: Save hyperparameters

            # TODO: Save figures

            # Save predictions
            # TODO: Implement
