import logging
from abc import ABC
from typing import Any, Dict, List, Optional, Union

import keras
import numpy as np
import optuna
import tensorflow as tf
from numpy import ndarray
from src.data_types.modules.lstm_keras_module import LstmKerasModule
from src.data_types.neural_net_model import NeuralNetModel
from src.pipelines import \
    local_univariate_lstm_keras_pipeline as lstm_keras_pipeline
from src.pipelines import local_univariate_lstm_pipeline as lstm_pipeline
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from src.save_experiment_source.local_checkpoint_save_source import \
    LocalCheckpointSaveSource
from src.utils.config_parser import config
from src.utils.keras_optimizer import KerasOptimizer
from src.utils.prettify_dict_string import prettify_dict_string


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
            pipeline=lstm_keras_pipeline.local_univariate_lstm_keras_pipeline,
        )
        self.should_shuffle_batches = params["should_shuffle_batches"]

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
        self.testing_data = tf.random.uniform(
            shape=(1, 7, 1), minval=0, maxval=1, dtype=tf.dtypes.float32, seed=16
        )
        self.test_labels = None
        """

    def process_data(self, data_set: Any, training_size: float) -> None:
        data_pipeline = self.pipeline(
            data_set=data_set,
            cat_id=self.get_name(),
            training_size=self.training_size,
            input_window_size=self.input_window_size,
            output_window_size=self.output_window_size,
        )
        logging.info(f"Data Pipeline for {self.get_name()}: {data_pipeline}")
        for log_source in self.log_sources:
            log_source.log_pipeline_steps(data_pipeline.__repr__())

        self.training_data, self.testing_data, self.min_max_scaler = data_pipeline.run()

    def init_neural_network(
        self, params: dict, logger=None, return_model: bool = False, **xargs
    ) -> Union[keras.Sequential, None]:
        model = LstmKerasModule(**params).model
        optim = KerasOptimizer.get(params["optimizer_name"], learning_rate=params["learning_rate"])

        model.compile(optimizer=optim, loss="mse")
        logging.info(
            f"Model compiled with optimizer {params['optimizer_name']}\n"
            f"{prettify_dict_string(params)}"
        )
        if return_model:
            return model
        else:
            self.model = model

    def train(self, epochs: int = None, **xargs) -> Dict:
        logging.info("Training")
        examples_to_drop_to_make_all_batches_same_size = (
            self.training_data[0].shape[0] % self.batch_size
        )
        logging.info(
            f"Examples to drop to make all batches same size: {examples_to_drop_to_make_all_batches_same_size}"
        )
        x_train, y_train = (
            self.training_data[0][:-examples_to_drop_to_make_all_batches_same_size],
            self.training_data[1][:-examples_to_drop_to_make_all_batches_same_size],
        )
        x_val, y_val = (
            x_train[-self.batch_size :],
            y_train[-self.batch_size :],
        )
        history = self.model.fit(
            x=x_train,
            y=y_train,
            epochs=self.number_of_epochs,
            batch_size=self.batch_size,
            shuffle=self.should_shuffle_batches,
            validation_data=(x_val, y_val),
        )
        self._copy_trained_weights_to_model_with_different_batch_size()
        history = history.history
        training_predictions = self.predict(x_train)
        validation_predictions = self.predict(x_val)
        # Visualize
        self._visualize_predictions(
            (y_train[:, 0].flatten()),
            (training_predictions[:, 0].flatten()),
            "Training predictions",
        )
        self._visualize_predictions(
            y_val[:, 0, 0],
            validation_predictions[:, 0],
            "Validation predictions",
        )
        self._visualize_errors(
            [history["loss"], history["val_loss"]], ["Training_errors", "Validation_errors"]
        )

        self.metrics["training_error"] = history["loss"][-1]
        self.metrics["validation_error"] = history["val_loss"][-1]
        return self.metrics

    def _copy_trained_weights_to_model_with_different_batch_size(self) -> None:
        trained_weights = self.model.get_weights()
        params = self.hyper_parameters
        params["batch_size"] = 1
        prediction_model = self.init_neural_network(params=params, return_model=True)
        prediction_model.set_weights(trained_weights)
        self.prediction_model = prediction_model

    def predict(self, input_data: ndarray, model: keras.Model = None) -> ndarray:
        logging.info("Predicting")
        return self.prediction_model.predict(input_data, batch_size=1)

    def test(self, predictive_period: int = 7, single_step: bool = False) -> Dict:
        logging.info("Testing")
        x_test, y_test = self.testing_data[0], self.testing_data[1]
        results = self.prediction_model.evaluate(x_test, y_test, batch_size=1)

        # Visualize
        test_predictions = self.prediction_model.predict(x_test, batch_size=1)
        self._visualize_predictions(
            tf.reshape(y_test, (-1,)),
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
        """
        logging.info("Tuning model")
        title, _ = LocalCheckpointSaveSource.load_title_and_description()
        study_name = f"{title}_{self.get_name()}"
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(),
            storage=
            # TODO: IModel should not rely on the config. Fix this
            f"sqlite:///{config['experiment']['save_source']['disk']['model_save_location'].get()}/optuna-tuning.db"
            if len(self.log_sources) > 0
            else None,
            load_if_exists=True,
        )
        logging.info(
            f"Loading or creating optuna study with name: {study_name}\n"
            f"Number of previous Trials with this name are #{len(study.trials)}"
        )
        parameter_space = parameters
        study.optimize(
            lambda trial: local_univariate_lstm_objective(
                trial=trial,
                hyperparameter_tuning_range=parameter_space,
                model=self,
            ),
            # TODO: Fix pytorch network to handle concurrency
            # n_jobs=8,  # Use maximum number of cores
            n_trials=parameter_space["number_of_trials"],
            show_progress_bar=False,
            callbacks=[self.log_trial],
        )
        id = f"{self.get_name()},{study.best_trial.number}"
        best_params = study.best_trial.params
        logging.info("Best params!", best_params)
        test_params = self.hyper_parameters.copy()
        test_params.update(best_params)
        logging.info("Params updated with best params", test_params)
        self.init_neural_network(test_params)
        best_score = study.best_trial.value
        logging.info(
            f"Best trial: {id}\n" f"best_score: {best_score}\n" f"best_params: {best_params}"
        )
        self._generate_optuna_plots(study)

        return {id: {"best_score": best_score, "best_params": best_params}}
        # TODO: Tune model
        pass
        """

    def get_model(self):
        return self.model

    def save(self, path: str) -> str:
        save_path = f"{path}{self.get_name}.h5"
        self.model.save_weights(save_path)
        return save_path

    def load(self, path: str) -> None:
        load_path = f"{path}{self.get_name}.h5"
        self.model.load_weights(load_path)
