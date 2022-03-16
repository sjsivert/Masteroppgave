import logging
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


class LstmModule(nn.Module):
    def __init__(
        self,
        device: str,
        input_window_size: int,
        number_of_features: int,
        hidden_layer_size: int,
        output_size: int,
        num_layers: int,
        learning_rate: float,
        batch_first: bool,
        dropout: float,
        bidirectional: bool = False,
    ):
        # Init pytorch module
        super(LstmModule, self).__init__()
        self.device = device

        # Model Parameters
        self.output_size = output_size  # shape of output
        self.num_layers = num_layers  # number of layers
        self.number_of_features = number_of_features  # number of features in each sample
        self.hidden_size = hidden_layer_size  # hidden state
        self.learning_rate = learning_rate  # learning rate
        self.dropout = nn.Dropout(p=dropout)

        self.lstm = nn.LSTM(
            input_size=number_of_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fully_connected_layer = nn.Linear(
            in_features=self.hidden_size, out_features=self.output_size * self.number_of_features
        )

        if self.device == "cuda":
            self.cuda()
            self.criterion.cuda()

    def reset_hidden_state(self, batch_size: int, x):
        logging.info("\n\n\n\n\n\n-----------------")
        logging.info("x device", x.device)
        # Here you have defined the hidden state, and internal state first, initialized with zeros.
        self.h_0 = Variable(
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        )
        self.c_0 = Variable(
            torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        )

        # Put tensors on same device as input
        self.h_0 = self.h_0.type_as(x)
        self.c_0 = self.c_0.type_as(x)
        print("hidden device", self.h_0.device)

    def forward(self, x):
        self.reset_hidden_state(x.size(0), x=x)
        # output (seq_len, batch, hidden_size * num_directions): tensor containing the output features (h_t) from the last layer of the RNN, for each t.
        # h_n (num_layers * num_directions, batch, hidden_size): tensor containing the hidden state for t=seq_len
        # c_n (num_layers * num_directions, batch, hidden_size): tensor containing the cell state for t=seq_len
        ula, (h_out, _) = self.lstm(x, (self.h_0, self.c_0))
        # Choose the hidden state from the last layer
        last_hidden_state_layer = h_out[-1]
        out = self.fully_connected_layer(last_hidden_state_layer)
        out = self.dropout(out)
        out = out.view(-1, self.output_size, self.number_of_features)
        """
        ___ Reshape for multi variate support ___
        In order to support the use of multi variate input,
        the output of the last hidden layer is outputting data as a multiple of features and time steps.
        It can then be reshaped to fit properly.
        
        The alternative is to reshape before the last layer, then use multi dim input on the last layer sized down.
        TODO! Evaluate what should be done of these two approaches
        """

        return out
