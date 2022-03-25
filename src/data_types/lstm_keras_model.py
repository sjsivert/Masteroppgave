import logging
from abc import ABC
from typing import List, Dict, Optional, Any, Union

import keras
import optuna
from numpy import ndarray

from src.data_types.modules.lstm_keras_module import LstmKerasModule
from src.data_types.neural_net_model import NeuralNetModel
from src.save_experiment_source.i_log_training_source import ILogTrainingSource

import tensorflow as tf
from src.pipelines import local_univariate_lstm_pipeline as lstm_pipeline
from src.utils.keras_optimizer import KerasOptimizer


class LstmKerasModel(NeuralNetModel, ABC):
    def __init__(
        self,
        log_sources: List[ILogTrainingSource],
        time_series_id: str,
        params: Dict,
        optuna_trial: Optional[optuna.trial.Trial] = None,
    ):
        super(LstmKerasModel, self).__init__(
            log_sources,
            time_series_id,
            params,
            optuna_trial,
            pipeline=lstm_pipeline.local_univariate_lstm_pipeline,
        )

        # Encoder contains a list of dicts or lists. Each with layer type, activation function if any, given a conv, number of filters and filter size
        # Encoder config
        # Placeholder data
        """
        self.training_data = tf.random.uniform(
            shape=(100, 7, 1), minval=0, maxval=1, dtype=tf.dtypes.float32, seed=14
        )
        self.training_labels = None
        self.validation_data = tf.random.uniform(
            shape=(1, 7, 1), minval=0, maxval=1, dtype=tf.dtypes.float32, seed=15
        )
        self.validation_lavels = None
        self.test_data = tf.random.uniform(
            shape=(1, 7, 1), minval=0, maxval=1, dtype=tf.dtypes.float32, seed=16
        )
        self.test_labels = None
        """

    def process_data(self, data_set: Any, training_size: float) -> None:
        pass

    def init_neural_network(
        self, params: dict, logger=None, return_model: bool = False, **xargs
    ) -> Union[keras.Sequential, None]:
        model = LstmKerasModule(**params).model
        optim = KerasOptimizer.get(params["optimizer_name"], learning_rate=params["learning_rate"])

        model.compile(optimizer=optim, loss="mse", metrics=["mse"])
        logging.info(
            f"Model compiled with optimizer {params['optimizer_name']}\n" f"{model.summary()}"
        )
        if return_model:
            return model
        else:
            self.model = model

    def train(self, epochs: int = None, **xargs) -> Dict:
        x_train, y_train = self.training_data[0], self.training_data[1]
        print(x_train.shape)
        print(y_train.shape)

        logging.info("Training")
        # Training the Auto encoder
        history = self.model.fit(
            x=x_train,
            y=y_train,
            epochs=self.number_of_epochs,
            batch_size=self.batch_size,
            shuffle=True,
            validation_data=(self.validation_data[0], self.validation_data[1]),
        )
        history = history.history
        training_predictions = self.predict(x_train)
        validation_predictions = self.predict(x_train)
        # Visualize
        self._visualize_predictions(
            tf.reshape(x_train, (-1,)),
            tf.reshape(training_predictions, (-1,)),
            "Training predictions",
        )
        self._visualize_predictions(
            tf.reshape(x_train, (-1,)),
            tf.reshape(x_train, (-1,)),
            "Validation predictions",
        )
        self._visualize_errors(history["loss"], history["val_loss"])

        self.metrics["training_error"] = history["loss"][-1]
        self.metrics["validation_error"] = history["val_loss"][-1]
        return self.metrics

    def predict(self, input_data: ndarray) -> ndarray:
        trained_weights = self.model.get_weights()
        params = self.hyper_parameters
        params["batch_size"] = 1
        prediction_model = self.init_neural_network(params=params, return_model=True)
        prediction_model.set_weights(trained_weights)
        return prediction_model.predict(input_data, batch_size=1)

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
