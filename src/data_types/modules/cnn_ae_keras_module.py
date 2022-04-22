import tensorflow as tf
from keras.layers import (
    Conv1D,
    Conv1DTranspose,
    MaxPool1D,
    Dropout,
    BatchNormalization,
    LayerNormalization,
    Flatten,
    Dense,
    Reshape,
)
from typing import List, Dict


def CNN_AE_Module(encoder_config: List[Dict] = None, decoder_config: List[Dict] = None):
    model = tf.keras.Sequential()
    # Encoder layers
    for e in encoder_config:
        add_layers(model, e)
        add_normalization_and_dropout(model, e)
    # Decoder layers
    for e in decoder_config:
        add_layers(model, e)
        add_normalization_and_dropout(model, e)
    return model


def add_layers(model, e):
    if e["layer"] == "Conv1d":
        model.add(
            Conv1D(
                e["filters"],
                e["kernel_size"],
                activation=(e["activation"] if "activation" in e else None),
            )
        )
    elif e["layer"] == "MaxPool":
        model.add(
            MaxPool1D(
                pool_size=e["size"],
                strides=(e["strides"] if "strides" in e else None),
                padding=e["padding"],
            )
        )
    elif e["layer"] == "Conv1dTranspose":
        model.add(
            Conv1DTranspose(
                e["filters"],
                e["kernel_size"],
                activation=(e["activation"] if "activation" in e else None),
            )
        )
    elif e["layer"] == "Flatten":
        model.add(Flatten())
    elif e["layer"] == "Dense":
        model.add(
            Dense(
                e["size"],
                activation=(e["activation"] if "activation" in e else None),
            )
        )
    elif e["layer"] == "Reshape":
        model.add(
            Reshape(
                (
                    e["size"],
                    e["filters"],
                )
            )
        )


def add_normalization_and_dropout(model, e):
    if "dropout" in e:
        model.add(Dropout(e["dropout"]))
    if "normalization" in e:
        model.add(
            BatchNormalization(
                scale=False,
            )
        )
