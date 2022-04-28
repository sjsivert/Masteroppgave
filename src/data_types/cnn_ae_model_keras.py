import logging
from abc import ABC
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
import pytorch_lightning as pl
import tensorflow as tf
from src.data_types.modules.cnn_ae_keras_module import CNN_AE_Module
from src.data_types.neural_net_keras_model import NeuralNetKerasModel
from src.data_types.neural_net_model import NeuralNetModel
from src.optuna_tuning.local_univariate_lstm_keras_objecktive import (
    local_univariate_cnn_ae_keras_objective,
)
from src.pipelines import local_univariate_lstm_keras_pipeline as lstm_keras_pipeline
from src.pipelines import local_univariate_lstm_pipeline as lstm_pipeline
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from src.utils.keras_optimizer import KerasOptimizer
from src.utils.visuals import visualize_data_series


class CNNAEModel(NeuralNetKerasModel, ABC):
    def init_neural_network(self, params: dict, logger=None, **xargs) -> None:
        self.model = CNN_AE_Module(params["encoder"], params["decoder"])
        optim = KerasOptimizer.get(params["optimizer_name"], learning_rate=params["learning_rate"])
        self.model.compile(optimizer=optim, loss=params["loss"])

    def train(self, epochs: int = None, tuning: bool = False, **xargs) -> Dict:
        if not tuning:
            x_train = np.concatenate([self.x_train, self.x_val], axis=0)
        else:
            x_train = self.x_train
        logging.info("Training")
        # Training the Auto encoder
        history = self.model.fit(
            x=x_train,
            y=x_train,
            epochs=self.hyper_parameters["number_of_epochs"],
            batch_size=self.batch_size,
            shuffle=self.should_shuffle_batches,
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
        logging.info("Tuning model")
        title, _ = LocalCheckpointSaveSource.load_title_and_description()
        study_name = f"{title}_{self.get_name()}"

        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner(),
            # TODO: IModel should not rely on the config. Fix this
            storage=f"sqlite:///{config['experiment']['save_source']['disk']['model_save_location'].get()}/optuna-tuning.db"
            if len(self.log_sources) > 0
            else None,
            load_if_exists=True,
        )
        logging.info(
            f"Loading or creating optuna study with name: {study_name}\n"
            f"Number of previous Trials with this name are #{len(study.trials)}"
        )
        logging.info("Init and train CNN-AE model")
        self.init_autoencoder()
        self.train_auto_encoder()

        study.optimize(
            lambda trial: local_univariate_cnn_ae_keras_objective(
                trial=trial,
                hyperparameter_tuning_range=parameters,
                model=self,
            ),
            timeout=parameters.get("time_to_tune_in_minutes", None),
            # TODO: Fix pytorch network to handle concurrency
            # n_jobs=-1,  # Use maximum number of cores
            n_trials=parameters.get("number_of_trials", None),
            # show_progress_bar=False,
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

    def get_model(self):
        return self.model

    def save(self, path: str) -> str:
        save_path = f"{path}{self.get_name}.h5"
        self.model.save_weights(save_path)
        return save_path

    def load(self, path: str) -> None:
        load_path = f"{path}{self.get_name}.h5"
        self.model.load_weights(load_path)
