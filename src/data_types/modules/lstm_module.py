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
from src.save_experiment_source.i_log_training_source import ILogTrainingSource


class LstmModule(nn.Modul):
    def __init__(
        self,
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
        # Init pytorch module
        super(LstmModule, self).__init__()

        # Model Parameters
        self.output_size = output_size  # shape of output
        self.num_layers = num_layers  # number of layers
        self.number_of_features = number_of_features  # number of features in each sample
        self.hidden_size = hidden_window_size  # hidden state
        self.learning_rate = learning_rate  # learning rate
        self.dropout = nn.Dropout(p=dropout)

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.fully_connected_layer = nn.Linear(
            in_features=self.hidden_size, out_features=self.output_size
        )

        parameters = list(self.lstm.parameters()) + list(self.fully_conencted_layer.parameters())
        if torch.cuda.is_available():
            self.cuda()
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.criterion.cuda()

    def forward(self, x):
        # Here you have defined the hidden state, and internal state first, initialized with zeros.
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        # output (seq_len, batch, hidden_size * num_directions): tensor containing the output features (h_t) from the last layer of the RNN, for each t.
        # h_n (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t=seq_len
        # c_n (num_layers * num_directions, batch, hidden_size): tensor containing the cell state for t=seq_len
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))

        # Choose the hidden state from the last layer
        last_hidden_state_layer = h_out[-1]
        out = self.fully_conencted_layer(last_hidden_state_layer)
        out = self.dropout(out)
        return out
