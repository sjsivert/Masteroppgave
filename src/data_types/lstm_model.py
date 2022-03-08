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
from src.data_types.i_model import IModel
from src.data_types.modules.lstm_module import LstmModule
from src.optuna_tuning.loca_univariate_lstm_objective import local_univariate_lstm_objective
from src.pipelines import local_univariate_lstm_pipeline as lstm_pipeline
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from torch import nn
from torch.autograd import Variable
from src.utils.pytorch_error_calculations import *

from src.save_experiment_source.local_checkpoint_save_source import LocalCheckpointSaveSource
from src.utils.visuals import visualize_data_series


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

        self.training_data_loader = None
        self.validation_data_loader = None
        self.testing_data_loader = None
        self.min_max_scaler = None

        self.optuna_trial = optuna_trial

        self.init_neural_network(params)

    def init_neural_network(self, params: dict) -> None:
        # Creating LSTM module
        self.model = LstmModule(
            input_window_size=params["input_window_size"],
            number_of_features=params["number_of_features"],
            hidden_layer_size=params["hidden_layer_size"],
            output_size=params["output_window_size"],
            num_layers=params["number_of_layers"],
            learning_rate=params["learning_rate"],
            batch_first=True,
            dropout=params["dropout"],
            bidirectional=False,
            device=self.device,
        )

        # TODO! Error metric selection
        self.criterion = calculate_error
        self.optimizer = getattr(torch.optim, params["optimizer_name"])(
            self.model.parameters(), lr=params["learning_rate"]
        )
        # TODO: Make using scheduler vs learning rate an option
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        # self.optimizer, patience=500, factor=0.5, min_lr=1e-7, eps=1e-08
        # )
        if self.device == "cuda":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.criterion.cuda()

    def calculate_mean_score(self, losses: List[float]) -> float64:
        return np.mean(losses)

    def get_name(self) -> str:
        return self.name

    def train(self, epochs: int = 100) -> Dict:
        # Visualization
        train_error = []
        val_error = []
        training_targets = []
        training_predictions = []

        # Training
        self.model.train()
        for epoch in progress_bar(range(epochs)):
            batch_train_error = []
            batch_val_error = []
            for x_batch, y_batch in self.training_data_loader:
                # the dataset "lives" in the CPU, so do our mini-batches
                # therefore, we need to send those mini-batches to the
                # device where the model "lives"
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                loss, _yhat = self._train_step(x_batch, y_batch)
                # Visualize
                batch_train_error.append(loss)
                if epoch + 1 == epochs:
                    pass
                    # TODO! Add support for multi step prediction visualization
                    # training_targets.extend(y_batch.reshape((y_batch.shape[0],)).tolist())
                    # training_predictions.extend(_yhat.reshape((_yhat.shape[0],)).tolist())

            epoch_train_error = sum(batch_train_error) / len(batch_train_error)
            train_error.append(epoch_train_error)

            for x_val, y_val in self.validation_data_loader:
                x_val = x_val.to(self.device)
                y_val = y_val.to(self.device)
                val_loss = self._validation_step(x_val, y_val)
                batch_val_error.append(val_loss)
            epoch_val_error = sum(batch_val_error) / len(batch_val_error)
            val_error.append(epoch_val_error)

            if self.optuna_trial:
                accuracy = self.calculate_mean_score(batch_val_error)
                self.optuna_trial.report(accuracy, epoch)

                if self.optuna_trial.should_prune():
                    print("Pruning trial!")
                    raise optuna.exceptions.TrialPruned()
            if (epoch + 1) % 50 == 0:
                logging.info(
                    f"Epoch: {epoch+1}, Training loss: {epoch_train_error}. Validation loss: {epoch_val_error}"
                )
        # TODO: Log historic data to continue training
        # self.metrics["training"] = train_error
        # self.metrics["validation"] = val_error

        self.metrics["training_error"] = train_error[-1]
        self.metrics["validation_error"] = val_error[-1]
        self._visualize_training(training_targets, training_predictions)
        self._visualize_training_errors(training_error=train_error, validation_error=val_error)
        return self.metrics

    # Builds function that performs a step in the train loop
    def _train_step(self, x, y) -> (float, torch.Tensor):
        # Make prediction, and compute loss, and gradients
        self.model.train()
        yhat = self.model(x)
        loss = self.criterion(y, yhat)
        loss.backward()
        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()
        # Returns the loss
        return loss.item(), yhat

    def _validation_step(self, x, y) -> float:
        error = None
        with torch.no_grad():
            self.model.eval()
            yhat = self.model(x)
            loss = self.criterion(y, yhat)
            error = loss.item()
        return error

    def _test_step(self, x, y) -> (Dict[str, float], torch.Tensor):
        with torch.no_grad():
            self.model.eval()
            yhat = self.model(x)
            # TODO! Add multi step support
            loss = calculate_errors(y, yhat)
        return loss, yhat

    def test(self, predictive_period: int = 6, single_step: bool = False) -> Dict:
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
            # testing_targets.extend(y_test.reshape((y_test.shape[0])).tolist())
            # testing_predictions.extend(_yhat.reshape((_yhat.shape[0])).tolist())
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

        for log_source in self.log_sources:
            log_source.log_pipeline_steps(data_pipeline.__repr__())

        (
            self.training_data_loader,
            self.validation_data_loader,
            self.testing_data_loader,
            self.min_max_scaler,
        ) = data_pipeline.run()

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
        best_score = study.best_trial.value
        logging.info(f"Best trial: {id}\n" f"best_score: {best_score}\n" f"best_params: {params}")
        return {id: {"best_score": best_score, "best_params": params}}

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
