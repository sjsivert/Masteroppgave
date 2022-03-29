import logging
from typing import Dict, List, Optional

from genpipes.compose import Pipeline
from pandas import DataFrame

from src.data_types.model_type_enum import ModelStructureEnum
from src.model_strutures.i_model_structure import IModelStructure
from src.model_strutures.local_univariate_arima_structure import \
    LocalUnivariateArimaStructure
from src.model_strutures.local_univariate_cnn_ae_lstm_structure import \
    LocalUnivariateCNNAELSTMStructure
from src.model_strutures.local_univariate_cnn_ae_structure import \
    LocalUnivariateCNNAEStructure
from src.model_strutures.local_univariate_lstm_structure import \
    LocalUnivariateLstmStructure
from src.model_strutures.validation_model_structure import \
    ValidationModelStructure
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from src.save_experiment_source.i_save_experiment_source import \
    ISaveExperimentSource
from src.save_experiment_source.local_checkpoint_save_source import (
    LocalCheckpointSaveSource, init_local_checkpoint_save_location)
from src.save_experiment_source.local_log_training_source import \
    LocalLogTrainingSource
from src.save_experiment_source.neptune_log_training_source import \
    NeptuneLogTrainingSource
from src.save_experiment_source.neptune_save_source import NeptuneSaveSource
from src.save_experiment_source.save_local_disk_source import \
    SaveLocalDiskSource
from src.utils.config_parser import config


class Experiment:
    """
    The main class for running experiments.
    It contains logic for choosing model type, logging results, and saving results.
    """

    def __init__(
        self,
        title: str = "",
        description: str = "",
        save_sources_to_use=[],
        save_source_options={},
        experiment_tags: Optional[List[str]] = [],
        load_from_checkpoint: bool = False,
        overwrite_save_location: bool = False,
    ) -> None:

        self.model_structure: IModelStructure = None
        self.title = title
        self.description = description
        self.experiment_description = description
        self.experiment_tags = experiment_tags
        self.save_sources = self._init_save_sources(
            save_sources_to_use,
            save_source_options,
            load_from_checkpoint,
            overwrite_save_location=overwrite_save_location,
        )

    def _init_save_sources(
        self,
        save_sources_to_use: List[str],
        save_source_options: Dict,
        load_from_checkpoint: bool,
        neptune_id_to_load: Optional[str] = None,
        overwrite_save_location: bool = False,
    ) -> List[ILogTrainingSource]:
        sources = []

        if not load_from_checkpoint:
            init_local_checkpoint_save_location(self.title, self.description)

        for source in save_sources_to_use:
            if source == "disk":
                sources.append(
                    LocalLogTrainingSource(
                        **save_source_options["disk"],
                        options_dump=config.dump(),
                        title=self.title,
                        description=self.experiment_description,
                        load_from_checkpoint=load_from_checkpoint,
                        overwrite_save_location=overwrite_save_location,
                    )
                )
            elif source == "neptune":
                sources.append(
                    NeptuneLogTrainingSource(
                        **save_source_options["neptune"],
                        title=self.title,
                        description=self.description,
                        load_from_checkpoint=load_from_checkpoint,
                        neptune_id_to_load=neptune_id_to_load,
                    )
                )

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
        logging.info("Saving experiment details before executing experiment")
        if save and options_to_save:
            for save_source in self.save_sources:
                save_source.save_experiment_details(
                    options=options_to_save,
                    datasets=config["data"].get(),
                    experiment_tags=self.experiment_tags,
                )
        logging.info(
            f"Running complete with parameters in specified in config. Experiment with saving set to {save}"
        )

        self._choose_model_structure(model_options=model_options)

        self._load_and_process_data(data_pipeline=data_pipeline)

        self._train_model()
        self._test_model()

        if save and options_to_save:
            self._save_model(options=options_to_save)

    def run_tuning_experiment(
        self,
        model_options: Dict,
        data_pipeline: Pipeline,
        save: bool = True,
        options_to_save: Optional[str] = None,
    ) -> None:
        """
        Run an experiment for auto-tuning of model and structure parameters
        """
        logging.info(f"Running tuning experiment with saving set to {save}")

        self._choose_model_structure(model_options=model_options)

        self._load_and_process_data(data_pipeline=data_pipeline)

        self.model_structure.auto_tuning()

        self._train_model()
        self._test_model()

        if save and options_to_save:
            self._save_model(options=options_to_save)

    def _choose_model_structure(self, model_options: Dict) -> IModelStructure:
        try:
            model_structure = ModelStructureEnum[model_options["model_type"]]
            if model_structure == ModelStructureEnum.validation_model:
                self.model_structure = ValidationModelStructure(self.save_sources)
            elif model_structure == ModelStructureEnum.local_univariate_arima:
                self.model_structure = LocalUnivariateArimaStructure(
                    self.save_sources, **model_options["local_univariate_arima"]
                )
            elif model_structure == ModelStructureEnum.local_univariate_lstm:
                self.model_structure = LocalUnivariateLstmStructure(
                    self.save_sources, **model_options["local_univariate_lstm"]
                )
            elif model_structure == ModelStructureEnum.local_cnn_ae:
                self.model_structure = LocalUnivariateCNNAEStructure(
                    self.save_sources, **model_options["local_univariate_cnn_ae"]
                )

            elif model_structure == ModelStructureEnum.local_cnn_ae_lstm:
                self.model_structure = LocalUnivariateCNNAELSTMStructure(
                    self.save_sources, **model_options["local_univariate_cnn_ae_lstm"]
                )

            logging.info(f"Choosing model structure: {self.model_structure}")
            self.model_structure.init_models()
            return self.model_structure

        except Exception as e:
            logging.error(
                f"Not a valid ModelType error: {e} \n \
                Valid ModelTypes are: {ModelStructureEnum.__members__}"
            )
            raise e

    def _load_and_process_data(self, data_pipeline: Pipeline) -> DataFrame:
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
            save_source.save_model_and_metadata(
                options=options,
                metrics=self.model_structure.get_metrics(),
                datasets=config["data"].get(),
                models=self.model_structure.get_models(),
                figures=self.model_structure.get_figures(),
                data_pipeline_steps=save_source.get_pipeline_steps(),
                experiment_tags=self.experiment_tags,
                tuning=self.model_structure.get_tuning_parameters(),
                predictions=self.model_structure.get_predictions(),
            )
