from abc import ABC
from typing import Optional, Dict, Tuple, List

import numpy
import numpy as np
import optuna
import torch
from fastprogress import progress_bar
from matplotlib.figure import Figure
from numpy import ndarray, float64
from pandas import DataFrame
from torch import nn
from torch.autograd import Variable

from src.data_types.i_model import IModel
from src.data_types.modules.lstm_module import LstmModule
from src.save_experiment_source.i_log_training_source import ILogTrainingSource


class LstmModel(IModel, ABC):
    def __init__(
        self,
        log_sources: List[ILogTrainingSource],
        name: str,
        input_window_size: int = 1,
        number_of_features: int = 1,
        hidden_window_size: int = 20,
        output_size: int = 1,
        num_layers: int = 1,
        learning_rate: float = 0.001,
        batch_first: bool = True,
        dropout: float = 0.2,
        bidirectional: bool = False,
        optimizer_name: str = "adam",
    ):
        self.log_sources: List[ILogTrainingSource] = log_sources
        self.name: str = name

        # Model Parameters
        self.output_size = output_size  # shape of output
        self.num_layers = num_layers  # number of layers
        self.number_of_features = number_of_features  # number of features in each sample
        self.hidden_size = hidden_window_size  # hidden state
        self.learning_rate = learning_rate  # learning rate
        self.dropout = nn.Dropout(p=dropout)

        self.criterion = nn.MSELoss()
        self.optimizer = getattr(torch.optim, optimizer_name)(
            self.parameters(), lr=learning_rate
        ).to(self.device)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=500, factor=0.5, min_lr=1e-7, eps=1e-08
        )

        # Creating LSTM module
        self.model = LstmModule(
            input_window_size=input_window_size,
            number_of_features=number_of_features,
            hidden_window_size=hidden_window_size,
            output_size=output_size,
            num_layers=num_layers,
            learning_rate=learning_rate,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            optimizer_name=optimizer_name,
        )

    def train_network(
        self,
        train_loader,
        val_loader,
        n_epochs=100,
        verbose=True,
        optuna_trial: optuna.Trial = None,
    ):
        losses = []
        val_losses = []
        for epoch in progress_bar(range(n_epochs)):
            for x_batch, y_batch in train_loader:
                # the dataset "lives" in the CPU, so do our mini-batches
                # therefore, we need to send those mini-batches to the
                # device where the model "lives"
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                loss = self.train_step(x_batch, y_batch)
                losses.append(loss)

            for x_val, y_val in val_loader:
                x_val = x_val.to(self.device)
                y_val = y_val.to(self.device)

                val_loss = self.validation_step(x_val, y_val)
                val_losses.append(val_loss)

            if optuna_trial:
                accuracy = self.calculate_mean_score(val_losses)
                optuna_trial.report(accuracy, epoch)

                if optuna_trial.should_prune():
                    print("Pruning trial!")
                    raise optuna.exceptions.TrialPruned()

            if epoch % 50 == 0:
                print(f"Epoch: {epoch}, loss: {loss}. Validation losses: {val_loss}")
        return losses, val_losses

    def calculate_mean_score(self, losses: ndarray) -> float64:
        return np.mean(losses)

    # Builds function that performs a step in the train loop
    def _train_step(self, x, y):
        self.train()  # Sets model to TRAIN mode

        yhat = self.model(x)  # Makes predictions
        loss = self.criterion(y, yhat)  # Computes loss
        loss.backward()  # Computes gradients
        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()
        # Returns the loss
        return loss.item()

    def _validation_step(self, x, y):
        val_losses = []
        with torch.no_grad():
            self.eval()

            yhat = self(x)
            val_loss = self.criterion(y, yhat)
            val_losses.append(val_loss.item())

        return val_losses

    def get_name(self) -> str:
        return self.name

    def process_data(
        self, data_set: DataFrame, training_size: float
    ) -> Tuple[DataFrame, DataFrame]:
        raise NotImplementedError()

    def train(self, epochs: int = 10) -> Dict:
        raise NotImplementedError()

    def test(self, predictive_period: int = 6, single_step: bool = False) -> Dict:
        raise NotImplementedError()

    def method_evaluation(
        self,
        parameters: List,
        metric: str,
        singe_step: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate model and calculate error of predictions for used for tuning evaluation
        """
        raise NotImplementedError()

    def get_figures(self) -> List[Figure]:
        """
        Return a list of figures created by the model for visualization
        """
        raise NotImplementedError()

    def get_metrics(self) -> Dict:
        """
        Fetch metrics from model training or testing
        """
        raise NotImplementedError()

    def save(self, path: str) -> str:
        """
        Save the model to the specified path.
        :returns: Path to saved model file
        """
        raise NotImplementedError()

    def load(self, path: str) -> IModel:
        """
        Load the model from the specified path.
        """
        raise NotImplementedError()

    def get_predictions(self) -> Optional[Dict]:
        """
        Returns the predicted values if test() has been called.
        """
        raise NotImplementedError()
