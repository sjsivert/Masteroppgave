import tensorflow as tf


class CNN_AE_LSTM_Module(tf.keras.Model):
    def __init__(self, ae: tf.keras.Model = None, lstm: tf.keras.Model = None):
        super().__init__()
        self.ae = ae
        self.lstm = lstm
        for layer in self.ae.layers:
            layer.trainable = False

    def call(self, inputs):
        ae_data = self.ae(inputs)
        out = self.lstm(ae_data)
        return out
