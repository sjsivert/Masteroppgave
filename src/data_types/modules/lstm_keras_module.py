import logging

from keras import Sequential
from keras.layers import LSTM, Dense
from numpy import ndarray

from src.utils.keras_optimizer import KerasOptimizer


class LSTMKeras:
    def __init__(
        self,
        optimizer_name: str,
        stateful: bool,
        output_window_size: int,
        number_of_lstm_layers: int,
        batch_training_size: int,
        input_window_size: int,
        number_of_features: int,
        hidden_layer_size: int,
        learning_rate: float,
        dropout: float = 0.0,
        recurrent_dropout=0.0,
        **kwargs,
    ):

        self.model = Sequential()
        for layer in range(number_of_lstm_layers):
            true_if_layer_is_not_last = True if layer < number_of_lstm_layers - 1 else False
            self.model.add(
                LSTM(
                    batch_input_shape=(batch_training_size, input_window_size, number_of_features),
                    units=hidden_layer_size,
                    return_sequences=true_if_layer_is_not_last,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    kernel_initializer="glorot_uniform",
                    recurrent_initializer="orthogonal",
                    bias_initializer="zeros",
                    stateful=False,
                )
            )
        self.model.add(
            Dense(
                units=output_window_size,
                activation="linear",
            )
        )

        optim = KerasOptimizer.get(optimizer_name, learning_rate=learning_rate)

        # TODO: Fix error functions
        self.model.compile(optimizer=optim, loss="mse", metrics=["mse"])

        logging.info(f"LSTM Keras model created\n{self.model.summary()}")

    def train(self, x_train: ndarray, y_train: ndarray, epochs: int, batch_size: int, verbose=1):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
