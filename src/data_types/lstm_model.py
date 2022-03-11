import logging
from abc import ABC
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import optuna
import pandas
import torch
from fastprogress import progress_bar
from matplotlib.figure import Figure
from numpy import float64, ndarray
from optuna import Study
from optuna.trial import FrozenTrial
from pandas import DataFrame
from torch.utils.data import DataLoader, Dataset

from src.data_types.i_model import IModel
from src.data_types.modules.lstm_lightning_module import LSTM_Lightning
from src.data_types.modules.lstm_module import LstmModule
from src.optuna_tuning.loca_univariate_lstm_objective import local_univariate_lstm_objective
from src.pipelines import local_univariate_lstm_pipeline as lstm_pipeline
from src.pipelines.simpe_time_series_pipeline import simple_time_series_pipeline
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from torch import nn
from torch.autograd import Variable
from src.utils.pytorch_error_calculations import *
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

from src.save_experiment_source.local_checkpoint_save_source import LocalCheckpointSaveSource
from src.utils.visuals import visualize_data_series

import pytorch_lightning as pl
from torch.nn import functional as F


class LstmModel(IModel, ABC):
    def __init__(
        self,
        log_sources: List[ILogTrainingSource],
        time_series_id: str,
        params: Dict,
        optuna_trial: Optional[optuna.trial.Trial] = None,
    ):

        # nn.MSELoss()
        # Init global variables
        self.model = None
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

        self.training_data_loader = None
        self.validation_data_loader = None
        self.testing_data_loader = None
        self.min_max_scaler = None

        self.optuna_trial = optuna_trial

        self.hyper_parameters = params

        logging.info("Running model on device: {}".format(self.device))

        self.init_neural_network(params)

    def init_neural_network(self, params: dict) -> None:
        # Creating LSTM module
        self.model = LSTM_Lightning()

    def calculate_mean_score(self, losses: List[float]) -> float64:
        return np.mean(losses)

    def get_name(self) -> str:
        return self.name

    def train(self, epochs: int = None) -> Dict:
        # Visualization
        training_targets = []
        training_predictions = []

        trainer = pl.Trainer(max_epochs=10)
        model = LSTM_Lightning()
        trainer.fit(model, train_dataloaders=self.training_data_loader)

        # Test loop
        train_error = []
        self.model.freeze()
        for x, y in self.training_data_loader:
            y_hat = self.model(x)
            loss = F.mse_loss(y_hat, y)
            train_error.append(loss.item())
            training_targets.extend(y.reshape(y.size(0)).tolist())
            training_predictions.extend(y_hat.reshape(y.size(0)).tolist())
        train_error = sum(train_error) / len(train_error)

        self.metrics["training_error"] = 0
        self._visualize_training(training_targets, training_predictions)
        return self.metrics

    def _test_step(self, x, y) -> (Dict[str, float], torch.Tensor):
        with torch.no_grad():
            self.model.eval()
            yhat = self.model(x)
            # TODO! Add multi step support
            loss = calculate_errors(y, yhat)
        return loss, yhat

    def test(self, predictive_period: int = 6, single_step: bool = False) -> Dict:
        return {}
        # Visualize
        testing_targets = []
        testing_predictions = []
        batch_test_error = []

        for x_test, y_test in self.testing_data_loader:
            x_test = x_test.to(self.device)
            y_test = y_test.to(self.device)
            test_loss, _yhat = self._test_step(x_test, y_test)
            # Visualization
            batch_test_error.append(test_loss)
            # TODO! Add support for multi step prediction visualization
            testing_targets.extend(y_test.reshape((y_test.shape[0])).tolist())
            testing_predictions.extend(_yhat.reshape((_yhat.shape[0])).tolist())
        batch_test_error_dict = {}
        for key in batch_test_error[0].keys():
            batch_test_error_dict[key] = sum([x[key] for x in batch_test_error]) / len(
                batch_test_error
            )
        logging.info(f"Testing error: {batch_test_error_dict}.")
        self.metrics.update(batch_test_error_dict)
        self._visualize_test(testing_targets, testing_predictions)

        return batch_test_error_dict

    def get_name(self) -> str:
        return self.name

    def process_data(self, data_set: DataFrame, training_size: float) -> None:
        data_pipeline = lstm_pipeline.local_univariate_lstm_pipeline(
            data_set=data_set,
            cat_id=self.get_name(),
            training_size=self.training_size,
            batch_size=self.batch_size,
            input_window_size=self.input_window_size,
            output_window_size=self.output_window_size,
        )
        data_pipeline = simple_time_series_pipeline()

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
            dataset=training_set, batch_size=batch_size, shuffle=should_shuffle
        )
        self.validation_data_loader = DataLoader(
            dataset=validation_set, batch_size=batch_size, shuffle=should_shuffle
        )
        self.testing_data_loader = DataLoader(
            dataset=testing_set, batch_size=batch_size, shuffle=should_shuffle
        )

    def log_trial(self, study: Study, trial: FrozenTrial) -> None:
        for log_source in self.log_sources:
            trial_info = trial.params
            trial_info["score"] = trial.value
            trial_info["Trial number"] = trial.number
            log_source.log_tuning_metrics(
                {f"{self.get_name():{trial.number}}": {"Parameters": trial_info}}
            )

    def method_evaluation(
        self,
        parameters: Any,
        metric: str,
        singe_step: bool = True,
    ) -> Dict[str, Dict[str, str]]:
        """
        Evaluate model and calculate error of predictions for used for tuning evaluation
        """
        title, _ = LocalCheckpointSaveSource.load_title_and_description()
        study_name = f"{title}_{self.get_name()}"
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(),
            storage=
            # TODO: IModel should not rely on the config. Fix this
            f"sqlite:///{config['experiment']['save_source']['disk']['model_save_location'].get()}/optuna-tuning.db"
            if len(self.log_sources) > 0
            else None,
            load_if_exists=True,
        )
        logging.info(
            f"Loading or creating optuna study with name: {study_name}\n"
            f"Number of previous Trials with this name are #{len(study.trials)}"
        )

        parameter_space = parameters
        study.optimize(
            lambda trial: local_univariate_lstm_objective(
                trial=trial,
                hyperparameter_tuning_range=parameter_space,
                metric_to_use_when_tuning=metric,
                model=self,
            ),
            # TODO: Fix pytorch network to handle concurrency
            # n_jobs=2, # Use maximum number of cores
            n_trials=parameter_space["number_of_trials"],
            show_progress_bar=True,
            callbacks=[self.log_trial],
        )
        id = f"{self.get_name()},{study.best_trial.number}"
        params = study.best_trial.params
        print("Best params!", params)
        test_params = self.hyper_parameters.copy()
        print("Params updated with best params", test_params)
        self.init_neural_network(test_params)
        best_score = study.best_trial.value
        logging.info(f"Best trial: {id}\n" f"best_score: {best_score}\n" f"best_params: {params}")
        self._generate_optuna_plots(study)

        return {id: {"best_score": best_score, "best_params": params}}

    def _generate_optuna_plots(self, study: Study) -> None:
        # TODO: Currently getting error Figure has not attribute axes. Fix
        pass
        # self.figures.append(plot_slice(study))
        # self.figures.append(plot_edf(study))
        # self.figures.append(plot_intermediate_values(study))
        # self.figures.append(plot_optimization_history(study))
        # self.figures.append(plot_parallel_coordinate(study))
        # self.figures.append(plot_param_importances(study))
        # self.figures.append(plot_slice(study))

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
        save_path = f"{path}LSTM_{self.get_name()}.pt"
        torch.save(self.model.state_dict(), save_path)
        return save_path

    def load(self, path: str) -> IModel:
        """
        Load the model from the specified path.
        The correct model should already be created as self.model.
        The load function only sets the pretrained verctor values from the saved model.
        """
        load_path = f"{path}LSTM_{self.get_name()}.pt"
        self.model.load_state_dict(torch.load(load_path))
        return self

    def get_predictions(self) -> Optional[Dict]:
        """
        Returns the predicted values if test() has been called.
        """
        raise NotImplementedError()

    def _visualize_training_errors(
        self, training_error: List[float], validation_error: List[float]
    ) -> None:
        # Visualize training and validation loss
        self.figures.append(
            visualize_data_series(
                title=f"{self.get_name()}# Training and Validation error",
                data_series=[training_error, validation_error],
                data_labels=["Training error", "Validation error"],
                colors=["blue", "orange"],
                x_label="Epoch",
                y_label="Error",
            )
        )

    def _visualize_training(self, targets, predictions):
        self.figures.append(
            visualize_data_series(
                title=f"{self.get_name()}# Training set fit",
                data_series=[targets, predictions],
                data_labels=["Training targets", "Training predictions"],
                colors=["blue", "orange"],
                x_label="Time",
                y_label="Interest",
            )
        )

    def _visualize_validation(self, targets, predictions):
        self.figures.append(
            visualize_data_series(
                title=f"{self.get_name()}# Validation set fit",
                data_series=[targets, predictions],
                data_labels=["Validation targets", "Validation predictions"],
                colors=["blue", "orange"],
                x_label="Time",
                y_label="Interest",
            )
        )

    def _visualize_test(self, targets, predictions):
        self.figures.append(
            visualize_data_series(
                title=f"{self.get_name()}# Test set predictions",
                data_series=[targets, predictions],
                data_labels=["Test targets", "Test predictions"],
                colors=["blue", "orange"],
                x_label="Time",
                y_label="Interest scaled",
            )
        )
