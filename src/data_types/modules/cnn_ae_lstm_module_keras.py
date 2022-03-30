import tensorflow as tf


class CNN_AE_LSTM_Module(tf.keras.Model):
    def __init__(self, ae: tf.keras.Model = None, lstm: tf.keras.Model = None):
        super().__init__()
        self.ae = ae
        self.lstm = lstm

    def call(self, inputs):
        ae_data = self.ae(inputs, training=False)
        out = self.lstm(ae_data)
        return out
