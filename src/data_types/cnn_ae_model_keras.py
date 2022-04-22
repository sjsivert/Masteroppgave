import logging
from abc import ABC
from typing import List, Dict, Optional, Any

from src.data_types.neural_net_keras_model import NeuralNetKerasModel
from src.pipelines import local_univariate_lstm_keras_pipeline as lstm_keras_pipeline
from src.data_types.modules.cnn_ae_keras_module import CNN_AE_Module
from src.data_types.neural_net_model import NeuralNetModel
from src.save_experiment_source.i_log_training_source import ILogTrainingSource

import optuna
import pytorch_lightning as pl
from src.pipelines import local_univariate_lstm_pipeline as lstm_pipeline
from src.utils.keras_optimizer import KerasOptimizer
from src.utils.visuals import visualize_data_series
import tensorflow as tf


class CNNAEModel(NeuralNetKerasModel, ABC):
    def init_neural_network(self, params: dict, logger=None, **xargs) -> None:
        self.model = CNN_AE_Module(params["encoder"], params["decoder"])
        optim = KerasOptimizer.get(params["optimizer_name"], learning_rate=params["learning_rate"])
        self.model.compile(optimizer=optim, loss=params["loss"])

    def train(self, epochs: int = None, **xargs) -> Dict:
        logging.info("Training")
        # Training the Auto encoder
        history = self.model.fit(
            x=self.x_train,
            y=self.x_train,
            epochs=self.hyper_parameters["number_of_epochs"],
            batch_size=self.batch_size,
            shuffle=True,
            validation_data=(self.x_val, self.x_val),
        )
        history = history.history
        training_predictions = self.model.predict(self.x_train)
        validation_predictions = self.model.predict(self.x_val)
        # Visualize
        self._visualize_predictions(
            tf.reshape(self.x_train, (-1,)),
            tf.reshape(training_predictions, (-1,)),
            "Training predictions",
        )
        self._visualize_predictions(
            tf.reshape(self.x_val, (-1,)),
            tf.reshape(validation_predictions, (-1,)),
            "Validation predictions",
        )
        self._visualize_errors([history["loss"], history["val_loss"]])

        self.metrics["training_error"] = history["loss"][-1]
        self.metrics["validation_error"] = history["val_loss"][-1]
        return self.metrics

    def test(self, predictive_period: int = 7, single_step: bool = False) -> Dict:
        results = self.model.evaluate(self.x_test, self.x_test, batch_size=self.batch_size)
        # Visualize
        test_predictions = self.model.predict(self.x_test)
        self._visualize_predictions(
            tf.reshape(self.x_test, (-1,)),
            tf.reshape(test_predictions, (-1,)),
            "Test predictions",
        )
        self.metrics["test_error"] = results
        return self.metrics

    def method_evaluation(
        self,
        parameters: Any,
        metric: str,
        singe_step: bool = True,
    ) -> Dict[str, Dict[str, str]]:
        logging.info("Tuning CNN-AE model")
        # TODO: Tune model
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
