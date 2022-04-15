import logging
from typing import List, Dict, Optional, Any

import keras
from numpy import ndarray

from src.data_types.modules.cnn_ae_keras_module import CNN_AE_Module
from src.data_types.modules.cnn_ae_lstm_module_keras import CNN_AE_LSTM_Module
from src.data_types.modules.lstm_keras_module import LstmKerasModule
from src.data_types.neural_net_keras_model import NeuralNetKerasModel
from src.pipelines import local_univariate_lstm_pipeline as lstm_pipeline
import optuna
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
import tensorflow as tf

from src.utils.keras_error_calculations import (
    config_metrics_to_keras_metrics,
    generate_error_metrics_dict,
)
from src.utils.keras_optimizer import KerasOptimizer


class CNNAELSTMModel(NeuralNetKerasModel):
    # TODO: Update config so this method is not needed
    def order_config(self):
        lstm_params = self.hyper_parameters["lstm-shared"]
        lstm_params.update(self.hyper_parameters["lstm"])
        lstm_params["batch_size"] = self.hyper_parameters["batch_size"]
        self.hyper_parameters["lstm"] = lstm_params

    def init_neural_network(self, params: dict, logger=None, **xargs) -> None:
        self.order_config()
        # Init CNN-AE
        self.init_autoencoder()
        # Init LSTM
        self.init_autoencoder_and_lstm()

    def init_autoencoder(self):
        self.ae = CNN_AE_Module(self.hyper_parameters["encoder"], self.hyper_parameters["decoder"])
        optim = KerasOptimizer.get(
            self.hyper_parameters["ae"]["optimizer_name"], learning_rate=self.hyper_parameters["ae"]["learning_rate"]
        )
        self.ae.compile(optimizer=optim, loss=self.hyper_parameters["ae"]["loss"])

    def init_autoencoder_and_lstm(self):
        lstm = LstmKerasModule(**self.hyper_parameters["lstm"]).model
        # Init Model class for merginig the two models
        keras_metrics = config_metrics_to_keras_metrics()
        self.model = CNN_AE_LSTM_Module(self.ae, lstm)
        optim = KerasOptimizer.get(
            self.hyper_parameters["lstm"]["optimizer_name"], learning_rate=self.hyper_parameters["lstm"]["learning_rate"]
        )
        self.model.compile(optimizer=optim, loss=keras_metrics[0], metrics=[keras_metrics])

    def train(self, epochs: int = None, **xargs) -> Dict:
        logging.info("Training Autoencoder")
        # Training the Auto encoder
        ae_history = self.ae.fit(
            x=self.x_train,
            y=self.x_train,
            epochs=self.hyper_parameters["ae"]["epochs"],
            batch_size=self.batch_size,
            shuffle=True,
            validation_data=(self.x_val, self.x_val),
        )
        ae_history = ae_history.history

        # Training CNN-AE-LSTM
        logging.info("Training CNN-AE and LSTM model")
        history = self.model.fit(
            x=self.x_train,
            y=self.y_train,
            epochs=self.hyper_parameters["lstm"]["epochs"],
            batch_size=self.batch_size,
            shuffle=True,
            validation_data=(self.x_val, self.y_val),
        )
        history = history.history

        # Visualize
        training_predictions = self.model.predict(self.x_train, batch_size=self.batch_size)
        validation_predictions = self.model.predict(self.x_val, batch_size=self.batch_size)
        self._visualize_predictions(
            tf.reshape(self.y_train, (-1,)),
            tf.reshape(training_predictions, (-1,)),
            "Training predictions",
        )
        self._visualize_predictions(
            tf.reshape(self.y_val, (-1,)),
            tf.reshape(validation_predictions, (-1,)),
            "Validation predictions",
        )

        self._visualize_errors(
            [history["loss"], history["val_loss"], ae_history["loss"], ae_history["val_loss"]],
            ["Training loss", "Validation loss", "AE training loss", "AE validation loss"],
        )

        self.metrics["training_error"] = history["loss"][0]
        self.metrics["validation_error"] = history["val_loss"][0]
        return self.metrics

    def test(self, predictive_period: int = 7, single_step: bool = False) -> Dict:
        # Copy LSTM weights for different batch sizes
        self._copy_trained_weights_to_model_with_different_batch_size()

        logging.info("Testing CNN-AE model")
        # Test AE model
        ae_test_predictions = self.ae.predict(self.x_test, batch_size=1)
        self._visualize_predictions(
            tf.reshape(self.x_test, (-1,)),
            tf.reshape(ae_test_predictions, (-1,)),
            "AE Test predictions",
        )
        # Evaluate model
        results = self.model.evaluate(self.x_test, self.y_test, batch_size=1)
        test_metrics = generate_error_metrics_dict(results[1:])

        test_predictions = self.model.predict(self.x_test)
        self._visualize_predictions(
            tf.reshape(self.y_test, (-1,)),
            tf.reshape(test_predictions, (-1,)),
            "Test predictions",
        )
        self.metrics.update(test_metrics)
        return self.metrics

    def _copy_trained_weights_to_model_with_different_batch_size(self) -> None:
        trained_lstm_weights = self.model.lstm.get_weights()
        params = self.hyper_parameters["lstm"]
        params["batch_size"] = 1
        self.lstm = LstmKerasModule(**params).model
        self.lstm.set_weights(trained_lstm_weights)
        self.model = CNN_AE_LSTM_Module(self.ae, self.lstm)
        keras_metrics = config_metrics_to_keras_metrics()
        self.model.compile(
            optimizer=params["optimizer_name"], loss=keras_metrics[0], metrics=[keras_metrics]
        )

    def method_evaluation(
        self,
        parameters: Any,
        metric: str,
        singe_step: bool = True,
    ) -> Dict[str, Dict[str, str]]:
        logging.info("Tuning CNN-AE and LSTM model")
        # TODO!
        pass

    def get_model(self):
        return self.model

    def save(self, path: str) -> str:
        save_path = f"{path}{self.get_name}.h5"
        self.model.save_weights(save_path)
        return save_path

    def load(self, path: str) -> None:
        load_path = f"{path}{self.get_name}.h5"
        self.model.load_weights(load_path)
