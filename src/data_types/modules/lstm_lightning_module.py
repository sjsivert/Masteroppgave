import torch
from torch import nn
from torch.autograd import Variable
import pytorch_lightning as pl
from torch.nn import functional as F


class LSTMLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # Set parameters
        self.output_size = 1
        self.input_size = 1
        self.num_layers = 2
        self.num_features = 1
        self.hidden_size = 32
        lr = 1e-3
        dropout = 0.2

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
            in_features=self.hidden_size, out_features=self.output_size * self.num_features
        )

    def reset_hidden_state(self, batch_size: int):
        # Here you have defined the hidden state, and internal state first, initialized with zeros.
        self.h_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))
        self.c_0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size))

    def forward(self, x):
        self.reset_hidden_state(batch_size=x.size(0))
        ula, (h_out, _) = self.lstm(x, (self.h_0, self.c_0))
        h_out_last = h_out[-1]
        out = self.out_layer(h_out_last)
        out_multi_feature = out.reshape(out.size(0), self.output_size, self.num_features)
        return out_multi_feature

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        yhat = self(x)
        loss = F.mse_loss(y, yhat)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        yhat = self(x)
        loss = F.mse_loss(y, yhat)
        self.log("val_loss", loss)
        return loss
