import logging
from typing import List

import tensorflow as tf
from keras import Sequential
from keras.layers import LSTM, Dense
from numpy import ndarray
from src.utils.keras_optimizer import KerasOptimizer


class LstmKerasModule:
    def __init__(
        self,
        output_window_size: int,
        # number_of_layers: int,
        batch_size: int,
        input_window_size: int,
        layers: List,
        number_of_features: int,
        # hidden_layer_size: int,
        # learning_rate: float,
        stateful_lstm: bool,
        # dropout: List[float] = [0.0],
        # recurrent_dropout=0.0,
        **kwargs,
    ):
        super().__init__()

        self.model = Sequential()
        for layer in range(len(layers)):
            true_if_layer_is_not_last = True if layer < len(layers) - 1 else False
            self.model.add(
                LSTM(
                    batch_input_shape=(batch_size, input_window_size, number_of_features),
                    # input_shape=(input_window_size, number_of_features),
                    units=layers[layer]["hidden_size"],
                    return_sequences=true_if_layer_is_not_last,
                    dropout=layers[layer]["dropout"],
                    recurrent_dropout=layers[layer]["recurrent_dropout"],
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    kernel_initializer="glorot_uniform",
                    recurrent_initializer="orthogonal",
                    bias_initializer="zeros",
                    stateful=stateful_lstm,
                )
            )
        self.model.add(
            Dense(
                units=output_window_size,
                activation="linear",
            )
        )
