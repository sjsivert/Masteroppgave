import logging
from abc import ABC
from typing import List, Dict, Optional, Any

from src.data_types.modules.cnn_ae_keras_module import CNN_AE_Module
from src.data_types.neural_net_model import NeuralNetModel
from src.save_experiment_source.i_log_training_source import ILogTrainingSource

import optuna
import pytorch_lightning as pl
from src.pipelines import local_univariate_lstm_pipeline as lstm_pipeline
from src.utils.visuals import visualize_data_series
import tensorflow as tf


class CNNAEModel(NeuralNetModel):
    def __init__(
        self,
        log_sources: List[ILogTrainingSource],
        time_series_id: str,
        params: Dict,
        optuna_trial: Optional[optuna.trial.Trial] = None,
    ):
        super(CNNAEModel, self).__init__(
            log_sources,
            time_series_id,
            params,
            optuna_trial,
            pipeline=lstm_pipeline.local_univariate_lstm_pipeline,
        )
        # Placeholder data
        self.training_data = None
        self.training_labels = None
        self.validation_data = None
        self.validation_labels = None
        self.test_data = None
        self.test_labels = None

    def process_data(self, data_set: Any, training_size: float) -> None:
        self.training_data = tf.random.uniform(
            shape=(100, 7, 1), minval=0, maxval=1, dtype=tf.dtypes.float32, seed=14
        )
        self.validation_data = tf.random.uniform(
            shape=(1, 7, 1), minval=0, maxval=1, dtype=tf.dtypes.float32, seed=15
        )
        self.test_data = tf.random.uniform(
            shape=(1, 7, 1), minval=0, maxval=1, dtype=tf.dtypes.float32, seed=16
        )

    def init_neural_network(self, params: dict, logger=None, **xargs) -> None:
        self.model = CNN_AE_Module(params["encoder"], params["decoder"])
        self.model.compile(optimizer=params["optimizer_name"], loss=params["loss"])

    def train(self, epochs: int = None, **xargs) -> Dict:
        logging.info("Training")
        # Training the Auto encoder
        history = self.model.fit(
            x=self.training_data,
            y=self.training_data,
            epochs=self.number_of_epochs,
            batch_size=self.batch_size,
            shuffle=True,
            validation_data=(self.validation_data, self.validation_data),
        )
        history = history.history
        training_predictions = self.model.predict(self.training_data)
        validation_predictions = self.model.predict(self.validation_data)
        # Visualize
        self._visualize_predictions(
            tf.reshape(self.training_data, (-1,)),
            tf.reshape(training_predictions, (-1,)),
            "Training predictions",
        )
        self._visualize_predictions(
            tf.reshape(self.validation_data, (-1,)),
            tf.reshape(validation_predictions, (-1,)),
            "Validation predictions",
        )
        self._visualize_errors([history["loss"], history["val_loss"]])

        self.metrics["training_error"] = history["loss"][-1]
        self.metrics["validation_error"] = history["val_loss"][-1]
        return self.metrics

    def test(self, predictive_period: int = 7, single_step: bool = False) -> Dict:
        results = self.model.evaluate(self.test_data, self.test_data, batch_size=32)
        # Visualize
        test_predictions = self.model.predict(self.test_data)
        self._visualize_predictions(
            tf.reshape(self.test_data, (-1,)),
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
