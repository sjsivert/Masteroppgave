import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import optuna
import pandas
import pytorch_lightning as pl
import torch
from fastprogress import progress_bar
from matplotlib.figure import Figure
from numpy import float64, ndarray
from optuna import Study
from optuna.trial import FrozenTrial
from optuna.visualization import (
    plot_contour,
    plot_edf,
    plot_intermediate_values,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
)
from pandas import DataFrame
from pytorch_lightning.loggers import NeptuneLogger
from src.data_types.i_model import IModel
from src.data_types.modules.lstm_lightning_module import LSTMLightning
from src.data_types.modules.lstm_module import LstmModule
from src.optuna_tuning.loca_univariate_lstm_objective import local_univariate_lstm_objective
from src.pipelines import local_univariate_lstm_pipeline as lstm_pipeline
from src.pipelines.simpe_time_series_pipeline import simple_time_series_pipeline
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from src.save_experiment_source.local_checkpoint_save_source import LocalCheckpointSaveSource
from src.save_experiment_source.neptune_save_source import NeptuneSaveSource
from src.utils.pytorch_error_calculations import *
from src.utils.visuals import visualize_data_series
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


class NeuralNetModel(IModel, ABC):
    def __init__(
        self,
        log_sources: List[ILogTrainingSource],
        time_series_id: str,
        params: Dict,
        optuna_trial: Optional[optuna.trial.Trial] = None,
        pipeline: Any = None,
    ):
        # Init global variables
        self.model = None
        self.trainer = None
        self.figures: List[Figure] = []
        self.metrics: Dict = {}
        self.log_sources: List[ILogTrainingSource] = log_sources
        self.name = time_series_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = params["batch_size"]
        self.training_size = params["training_size"]
        self.input_window_size = params["input_window_size"]
        self.output_window_size = params["output_window_size"]
        self.number_of_epochs = params["number_of_epochs"]
        self.pipeline = pipeline

        self.training_data_loader = None
        self.validation_data_loader = None
        self.testing_data_loader = None
        self.min_max_scaler = None

        self.optuna_trial = optuna_trial

        self.hyper_parameters = params

        logging.info("Running model on device: {}".format(self.device))

        self.init_neural_network(params)

    @abstractmethod
    def init_neural_network(self, params: dict, logger=None, **xargs) -> None:
        return NotImplemented

    def _get_neptune_run_from_save_sources(self) -> Optional[NeptuneLogger]:
        for log_source in self.log_sources:
            if isinstance(log_source, NeptuneSaveSource):
                cast(NeptuneSaveSource, log_source)
                neptune_run = log_source.run
                logging.info("Using pre-existing neptune run for PytorchLightning")
                neptune_logger = NeptuneLogger(log_model_checkpoints=False, run=neptune_run)
                return neptune_logger

    def get_name(self) -> str:
        return self.name

    @abstractmethod
    def train(self, epochs: int = None, **xargs) -> Dict:
        return NotImplemented

    @abstractmethod
    def test(self, predictive_period: int = 6, single_step: bool = False) -> Dict:
        return NotImplemented

    def process_data(self, data_set: DataFrame, training_size: float) -> None:
        data_pipeline = self.pipeline(
            data_set=data_set,
            cat_id=self.get_name(),
            training_size=self.training_size,
            input_window_size=self.input_window_size,
            output_window_size=self.output_window_size,
        )
        logging.info(f"Data Pipeline for {self.get_name()}: {data_pipeline}")
        for log_source in self.log_sources:
            log_source.log_pipeline_steps(data_pipeline.__repr__())
        (
            self.training_dataset,
            self.validation_dataset,
            self.testing_dataset,
            self.min_max_scaler,
        ) = data_pipeline.run()
        self._convert_dataset_to_dataloader(
            self.training_dataset,
            self.validation_dataset,
            self.testing_dataset,
            batch_size=self.batch_size,
        )

    def _convert_dataset_to_dataloader(
        self,
        training_set: Dataset,
        validation_set: Dataset,
        testing_set: Dataset,
        batch_size: int,
        should_shuffle: bool = False,
    ) -> None:
        logging.info(f"Converting dataset to dataloader using batch size {batch_size}.")
        self.training_data_loader = DataLoader(
            dataset=training_set,
            batch_size=batch_size,
            shuffle=should_shuffle,
            num_workers=0,
        )
        self.validation_data_loader = DataLoader(
            dataset=validation_set,
            batch_size=1,
            shuffle=should_shuffle,
            num_workers=0,
        )
        self.testing_data_loader = DataLoader(
            dataset=testing_set,
            batch_size=1,
            shuffle=should_shuffle,
            num_workers=0,
        )

    def log_trial(self, study: Study, trial: FrozenTrial) -> None:
        for log_source in self.log_sources:
            trial_info = trial.params
            trial_info["score"] = trial.value
            trial_info["Trial number"] = trial.number
            log_source.log_tuning_metrics(
                {f"{self.get_name():{trial.number}}": {"Parameters": trial_info}}
            )

    @abstractmethod
    def method_evaluation(
        self,
        parameters: Any,
        metric: str,
        singe_step: bool = True,
    ) -> Dict[str, Dict[str, str]]:
        return NotImplemented

    def _generate_optuna_plots(self, study: Study) -> None:
        # TODO: Currently getting error Figure has not attribute axes. Fix
        self.figures.append(
            plot_slice(study).update_layout(title=f"{self.get_name()} - Plot Slice")
        )
        self.figures.append(plot_edf(study).update_layout(title=f"{self.get_name()} - Plot EDF"))
        self.figures.append(
            plot_intermediate_values(study).update_layout(
                title=f"{self.get_name()} - Plot Intermediate Values"
            )
        )
        self.figures.append(
            plot_optimization_history(study).update_layout(
                title=f"{self.get_name()} - Plot Optimization History"
            )
        )
        self.figures.append(
            plot_parallel_coordinate(study).update_layout(
                title=f"{self.get_name()} - Plot Parallel Coordinate"
            )
        )
        # Need multiple trials to plot this
        if len(study.trials) > 1:
            self.figures.append(
                plot_param_importances(study).update_layout(
                    title=f"{self.get_name()} - Plot Param Importances"
                )
            )

    def get_figures(self) -> List[Figure]:
        """
        Return a list of figures created by the model for visualization
        """
        return self.figures

    def get_metrics(self) -> Dict:
        """
        Fetch metrics from model training or testing
        """
        return self.metrics

    def save(self, path: str) -> str:
        """
        Save the model to the specified path.
        :returns: Path to saved model file
        """
        save_path = f"{path}{self.get_name()}.pt"
        torch.save(self.model.state_dict(), save_path)
        return save_path

    def load(self, path: str) -> IModel:
        """
        Load the model from the specified path.
        The correct model should already be created as self.model.
        The load function only sets the pretrained verctor values from the saved model.
        """
        load_path = f"{path}{self.get_name()}.pt"
        self.model.load_state_dict(torch.load(load_path))
        return self

    def get_predictions(self) -> Optional[Dict]:
        """
        Returns the predicted values if test() has been called.
        """
        raise NotImplementedError()

    def _visualize_errors(
        self, errors: List[List[float]], labels: List[str] = ["Training error", "Validation error"]
    ) -> None:
        # Visualize training and validation loss
        self.figures.append(
            visualize_data_series(
                title=f"{self.get_name()}# Training and Validation error",
                data_series=[x for x in errors],
                data_labels=labels,
                colors=["blue", "orange", "red", "green"],
                x_label="Epoch",
                y_label="Error",
            )
        )

    def _visualize_predictions(self, targets, predictions, name: str):
        self.figures.append(
            visualize_data_series(
                title=f"{self.get_name()}# {name}",
                data_series=[targets, predictions],
                data_labels=["Targets", "Predictions"],
                colors=["blue", "orange"],
                x_label="Time",
                y_label="Interest",
            )
        )
