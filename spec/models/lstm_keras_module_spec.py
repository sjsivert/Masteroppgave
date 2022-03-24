import numpy as np
from expects import expect, be_true, be, match, equal
from genpipes.compose import Pipeline
from mamba import description, it

from spec.utils.test_data import random_data_loader
from src.data_types.modules.lstm_keras_module import LSTMKeras

with description(LSTMKeras, "this") as self:
    with it("should be instantiable"):
        model = LSTMKeras(
            optimizer_name="Adam",
            stateful=False,
            output_window_size=1,
            number_of_lstm_layers=3,
            batch_training_size=32,
            input_window_size=1,
            number_of_features=1,
            hidden_layer_size=64,
            recurrent_dropout=0.0,
            learning_rate=0.001,
            dropout=0.0,
        )
