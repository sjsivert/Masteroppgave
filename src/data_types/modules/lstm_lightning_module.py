from typing import Tuple, List, Dict

import torch
from torch import nn, Tensor
from torch.autograd import Variable
import pytorch_lightning as pl
from torch.nn import functional as F


class LSTMLightning(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        learning_rate: float,
        batch_first: bool = True,
        dropout: float = 0.2,
        bidirectional: bool = False,
        optimizer_name: str = "Adam",
        batch_size: int = 1,
    ):
        super().__init__()

        self.output_size = output_size  # shape of output
        self.num_layers = num_layers  # number of layers
        self.input_size = input_size  # number of features in each sample
        self.hidden_size = hidden_size  # hidden state
        self.learning_rate = learning_rate  # learning rate
        self.dropout = nn.Dropout(p=dropout)
        self.batch_size = batch_size

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.fully_conencted_layer = nn.Linear(
            in_features=self.hidden_size, out_features=self.output_size * self.input_size
        )

    def reset_hidden_state(self, batch_size):

        self.h_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))

        self.c_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))

    def forward(self, x):
        self.reset_hidden_state(batch_size=x.size(0))
        # Here you have defined the hidden state, and internal state first, initialized with zeros.

        # Propagate input through LSTM

        # output (seq_len, batch, hidden_size * num_directions): tensor containing the output features (h_t) from the last layer of the RNN, for each t.
        # h_n (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t=seq_len
        # c_n (num_layers * num_directions, batch, hidden_size): tensor containing the cell state for t=seq_len
        ula, (h_out, _) = self.lstm(x, (self.h_0, self.c_0))
        # print("output", ula.shape)
        # Choose the hidden state from the last layer
        last_hidden_state_layer = h_out[-1]
        # h_out = h_out.view(-1, self.hidden_size)
        # out_squashed = ula.view(-1, self.hidden_size)

        out = self.fully_conencted_layer(last_hidden_state_layer)
        # out = self.fc(ula)
        # out = self.dropout(out)

        out_multi_feature = out.reshape(out.size(0), self.output_size, self.input_size)
        # return out
        return out_multi_feature

    def training_step(self, train_batch: Tuple[Tensor], batch_idx: int):
        x, y = train_batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, val_batch: Tuple[Tensor], batch_idx: int):
        x, y = val_batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, test_batch: Tuple[Tensor], batch_idx: int):
        x, y = test_batch
        y_pred = self(x)
        loss = F.mse_loss(y_pred, y)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)

    def training_epoch_end(self, training_step_outputs: List[Dict]):
        avg_train_loss = torch.stack([x["loss"] for x in training_step_outputs]).mean()
        self.log("avg_train_loss", avg_train_loss)
