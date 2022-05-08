import logging

import pytorch_lightning as pl
import torch
from src.utils.pytorch_error_calculations import calculate_error, calculate_errors
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader


class LSTMLightning(pl.LightningModule):
    def __init__(
        self,
        input_window_size: int,
        number_of_features: int,
        hidden_layer_size: int,
        output_window_size: int,
        number_of_layers: int,
        learning_rate: float,
        dropout: float,
        optimizer_name: str,
        **kwargs,
    ):
        super().__init__()

        # Set params for error progression
        self.training_errors = []
        self.validation_errors = []
        self.test_loss = 0

        self.test_losses = []
        self.test_losses_dict = {}
        # Set params for test prediction visualization
        self.test_targets = []
        self.test_predictions = []

        self.val_targets = []
        self.val_predictions = []

        # Set parameters
        self.output_window_size = output_window_size
        self.input_window_size = input_window_size
        self.num_layers = number_of_layers
        self.num_features = number_of_features
        self.hidden_size = hidden_layer_size
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name

        logging.info(
            f"Initializing LSTM with {self.num_layers} layers, {self.hidden_size} hidden size, "
            f"{self.learning_rate} learning rate, {dropout} dropout. "
            f"Optimiser: {self.optimizer_name},  "
            f"input window size: {self.input_window_size}, output window size: {self.output_window_size}"
        )

        # Metric (loss / error)
        self.metric = calculate_error
        # self.metric = nn.MSELoss()

        # LSTM model
        self.lstm = nn.LSTM(
            input_size=self.num_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout,
        )

        # Fully connected output layer
        self.out_layer = nn.Linear(
            in_features=self.hidden_size, out_features=self.output_window_size * self.num_features
        )

    def reset_hidden_state(self, batch_size: int, x):
        # Here you have defined the hidden state, and internal state first, initialized with zeros.
        self.h_0 = Variable(
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        )
        self.c_0 = Variable(
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        )

    def forward(self, x):
        self.reset_hidden_state(batch_size=x.size(0), x=x)
        ula, (h_out, _) = self.lstm(x, (self.h_0, self.c_0))
        h_out_last = h_out[-1]
        out = self.out_layer(h_out_last)
        out_multi_feature = out.reshape(out.size(0), self.output_window_size, self.num_features)
        return out_multi_feature

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer_name)(
            self.parameters(), lr=self.learning_rate
        )
        logging.info(f"Using optimizer: {optimizer.__class__.__name__}")
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        yhat = self(x)
        loss = self.metric(y, yhat)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, training_step_outputs):
        epoch_loss = []
        for out in training_step_outputs:
            batch_loss = out["loss"]
            epoch_loss.append(batch_loss)
        epoch_loss = sum(epoch_loss) / len(epoch_loss)
        self.log(
            "training_epoch_loss",
            epoch_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.training_errors.append(epoch_loss.cpu().detach().numpy().item())

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        yhat = self(x)
        loss = self.metric(y, yhat)

        self.val_targets.extend(y.flatten().tolist())
        self.val_predictions.extend(yhat.flatten().tolist())

        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_epoch_end(self, validation_step_outputs) -> None:
        epoch_loss = []
        for out in validation_step_outputs:
            epoch_loss.append(out)
        epoch_loss = sum(epoch_loss) / len(epoch_loss)
        self.validation_errors.append(epoch_loss.cpu().detach().numpy().item())

    def test_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        loss = self.metric(y, yhat)
        losses_dict = calculate_errors(y, yhat)
        self.test_losses.append(losses_dict)
        self.log("test_loss", loss)
        return loss

    def test_epoch_end(self, outputs):
        # Create dict containing mean of batch losses of all loss metrics
        for key in self.test_losses[0].keys():
            self.test_losses_dict[key] = sum([x[key] for x in self.test_losses]) / len(
                self.test_losses
            )
        # Store test loss
        test_loss = []
        for out in outputs:
            test_loss.append(out.item())
        self.test_loss = sum(test_loss) / len(test_loss)
        self.log("Test_loss", self.test_loss)

    def visualize_predictions(self, dataset: DataLoader):
        """
        Return selected targets and predictions for visualization of current predictive ability
        """
        targets = []
        predictions = []
        for batch_idx, batch in enumerate(dataset):
            x, y = batch
            y_hat = self.predict_step(x, batch_idx)
            targets.extend(y.detach().numpy().flatten())
            predictions.extend(y_hat.detach().numpy().flatten())
        return targets, predictions
