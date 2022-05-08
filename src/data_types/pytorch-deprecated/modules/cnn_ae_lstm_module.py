import logging
from typing import List

import pytorch_lightning as pl
import torch
from torch import Tensor
from ast import Tuple
from torch import nn
from torch.utils.data import DataLoader


class CNN_AE_LSTM(pl.LightningModule):
    def __init__(self, autoencoder: pl.LightningModule, lstm: pl.LightningModule):
        super().__init__()
        self.autoencoder = autoencoder
        self.lstm = lstm
        self.criterion = nn.MSELoss()

        # Visualization
        self.training_error: List[float] = []
        self.validation_error: List[float] = []
        self.test_error: float = None

    def forward(self, x):
        # Autoencoder x recreation
        x_hat = self.autoencoder.forward(x)
        # LSTM y prediction
        out = self.lstm.forward(x_hat)
        return out

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.lstm.parameters(), lr=0.001)
        return optim

    def training_step(self, train_batch: Tuple(Tensor), batch_idx: int):
        x, y = train_batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        return loss

    def validation_step(self, val_batch: Tuple(Tensor), batch_idx: int):
        x, y = val_batch
        y_hat = self.predict_step(x, batch_idx)
        loss = self.criterion(y_hat, y)
        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Auto encoder
        x, y = batch
        y_hat = self.predict_step(x, batch_idx)
        loss = self.criterion(y_hat, y)
        return loss

    def training_epoch_end(self, outputs):
        epoch_loss = []
        for out in outputs:
            batch_loss = out["loss"]
            epoch_loss.append(batch_loss)
        epoch_loss = sum(epoch_loss) / len(epoch_loss)
        self.training_error.append(epoch_loss.detach().item())

    def validation_epoch_end(self, outputs):
        epoch_loss = []
        for out in outputs:
            epoch_loss.append(out)
        epoch_loss = sum(epoch_loss) / len(epoch_loss)
        self.validation_error.append(epoch_loss.detach().item())

    def test_epoch_end(self, outputs):
        test_loss = []
        for out in outputs:
            test_loss.append(out.item())
        self.test_error = sum(test_loss) / len(test_loss)
        self.log("Test_loss", self.test_error)

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
