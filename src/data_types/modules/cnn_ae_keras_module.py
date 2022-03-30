import tensorflow as tf
from keras.layers import Conv1D, Conv1DTranspose
from typing import List, Dict


def CNN_AE_Module(encoder_config: List[Dict] = None, decoder_config: List[Dict] = None):
    model = tf.keras.Sequential()
    # Encoder layers
    for e in encoder_config:
        model.add(
            Conv1D(
                e["filters"],
                e["kernel_size"],
                activation=(e["activation"] if "activation" in e else None),
            )
        )
    # Decoder layers
    for e in encoder_config:
        model.add(
            Conv1DTranspose(
                e["filters"],
                e["kernel_size"],
                activation=(e["activation"] if "activation" in e else None),
            )
        )
    return model
