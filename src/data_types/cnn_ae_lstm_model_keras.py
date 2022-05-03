import logging
from typing import Any, Dict, List, Optional

import keras
import numpy as np
import optuna
import tensorflow as tf
from numpy import ndarray
from src.data_types.modules.cnn_ae_keras_module import CNN_AE_Module
from src.data_types.modules.cnn_ae_lstm_module_keras import CNN_AE_LSTM_Module
from src.data_types.modules.lstm_keras_module import LstmKerasModule
from src.data_types.neural_net_keras_model import NeuralNetKerasModel
from src.optuna_tuning.local_univariate_lstm_keras_objecktive import (
    local_univariate_cnn_ae_lstm_keras_objective,
)
from src.utils.lr_scheduler import scheduler
from src.utils.keras_error_calculations import (
    config_metrics_to_keras_metrics,
    generate_error_metrics_dict,
    keras_mase_periodic,
)
from src.pipelines import local_univariate_lstm_pipeline as lstm_pipeline
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from src.save_experiment_source.local_checkpoint_save_source import LocalCheckpointSaveSource
from src.utils.config_parser import config
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
        self.hyper_parameters["lstm"] = lstm_params.copy()

    def init_neural_network(self, params: dict, logger=None, **xargs) -> None:
        self.order_config()
        # Init CNN-AE
        self.init_autoencoder()
        # Init LSTM
        self.init_autoencoder_and_lstm(self.hyper_parameters["lstm"].copy())

    def init_autoencoder(self):
        self.ae = CNN_AE_Module(self.hyper_parameters["encoder"], self.hyper_parameters["decoder"])
        optim = KerasOptimizer.get(
            self.hyper_parameters["ae"]["optimizer_name"],
            learning_rate=self.hyper_parameters["ae"]["learning_rate"],
        )
        self.ae.compile(optimizer=optim, loss=self.hyper_parameters["ae"]["loss"])

    def init_autoencoder_and_lstm(
        self, params: dict, logger=None, return_model: bool = False, **xargs
    ):
        lstm = LstmKerasModule(**params).model
        # Init Model class for merginig the two models
        keras_metrics = config_metrics_to_keras_metrics()
        self.model = CNN_AE_LSTM_Module(self.ae, lstm)
        optim = KerasOptimizer.get(params["optimizer_name"], learning_rate=params["learning_rate"])
        self.model.compile(optimizer=optim, loss=keras_metrics[0], metrics=[keras_metrics])

    def train(self, epochs: int = None, **xargs) -> Dict:
        ae_metrics, ae_history = self.train_auto_encoder()
        lstm_ae_metrics, history = self.train_lstm(epochs=self.hyper_parameters["lstm"]["epochs"])
        self._visualize_errors(
            [history["loss"], history["val_loss"], ae_history["loss"], ae_history["val_loss"]],
            ["Training loss", "Validation loss", "AE training loss", "AE validation loss"],
        )
        self.metrics.update(lstm_ae_metrics)
        return self.metrics

    def train_auto_encoder(self, tuning=False, **xargs):
        if not tuning:
            x_train = np.concatenate([self.x_train, self.x_val], axis=0)
        else:
            x_train = self.x_train
        logging.info("Training Autoencoder")
        # Training the Auto encoder
        history = self.ae.fit(
            x=x_train,
            y=x_train,
            epochs=self.hyper_parameters["ae"]["epochs"],
            batch_size=self.batch_size,
            shuffle=True,  # self.should_shuffle_batches,
            validation_data=(self.x_val, self.x_val),
        )
        history = history.history
        return {
            "training_error": history["loss"][0],
            "validation_error": history["val_loss"][0],
        }, history

    def train_lstm(self, epochs=1, tuning=False, **xargs):
        if not tuning:
            x_train = np.concatenate([self.x_train, self.x_val], axis=0)
            y_train = np.concatenate([self.y_train, self.y_val], axis=0)
        else:
            x_train = self.x_train
            y_train = self.y_train
        # Training CNN-AE-LSTM
        logging.info("Training CNN-AE and LSTM model")
        callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=1)
        history = self.model.fit(
            x=x_train,
            y=y_train,
            epochs=epochs,
            batch_size=self.batch_size,
            shuffle=self.should_shuffle_batches,
            validation_data=(self.x_val, self.y_val),
            callbacks=[callback],
            **xargs,
        )
        history = history.history

        # Visualize
        if not tuning:
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
        return {
            "training_error": history["loss"][0],
            "validation_error": history["val_loss"][0],
        }, history

    def test(self, predictive_period: int = 7, single_step: bool = False) -> Dict:
        x_train = np.concatenate([self.x_train, self.x_val], axis=0)
        y_train = np.concatenate([self.y_train, self.y_val], axis=0)
        x_test, y_test = self.x_test, self.y_test
        # Copy LSTM weights for different batch sizes
        self._copy_trained_weights_to_model_with_different_batch_size()

        logging.info("Testing CNN-AE model")
        # Test AE model
        ae_test_predictions = self.ae.predict(x_test, batch_size=1)
        self._visualize_predictions(
            tf.reshape(x_test, (-1,)),
            tf.reshape(ae_test_predictions, (-1,)),
            "AE Test predictions",
        )

        # Evaluate model
        self.model.lstm.reset_states()
        results: List[float] = self.model.evaluate(
            x_train,
            y_train,
            batch_size=1,
        )
        results: List[float] = self.model.evaluate(
            x_test,
            y_test,
            batch_size=1,
        )
        # Remove first element because it is a duplication of the second element.
        test_metrics = generate_error_metrics_dict(results[1:])

        # Visualize
        # Copy LSTM weights for different batch sizes
        self.model.lstm.reset_states()
        self.predict_and_rescale(x_train, y_train)
        test_predictions, test_targets = self.predict_and_rescale(self.x_test, self.y_test)
        last_period_targets = (
            self.min_max_scaler.inverse_transform(x_test[:, -self.output_window_size :, 0])
            if self.min_max_scaler
            else x_test[:, -self.output_window_size :, 0]
        )
        mase_periode, y_true_last_period = keras_mase_periodic(
            y_true=test_targets, y_true_last_period=last_period_targets, y_pred=test_predictions
        )
        test_metrics[f"test_MASE_{len(self.x_test)}_DAYS"] = mase_periode.numpy()

        self._visualize_predictions(
            tf.reshape(test_targets, (-1,)),
            tf.reshape(test_predictions, (-1,)),
            "Test predictions",
        )
        self._visualize_predictions_and_last_period(
            (test_targets.flatten()),
            (test_predictions.flatten()),
            last_period_targets.flatten(),
            "Test predictions with last period targets",
        )
        x_test_values = (
            self.min_max_scaler.inverse_transform(x_test[:, :, 0])
            if self.min_max_scaler
            else x_test[:, :, 0]
        )
        x_test_values_flattened = x_test_values.flatten()
        self._visualize_predictions_with_context(
            context=x_test_values_flattened,
            targets=test_targets.flatten(),
            predictions=test_predictions.flatten(),
        )
        self.metrics.update(test_metrics)
        return self.metrics

    def _rescale_data(self, data: ndarray) -> ndarray:
        return self.min_max_scaler.inverse_transform(data)

    def predict_and_rescale(self, input_data: ndarray, targets: ndarray) -> ndarray:
        logging.info("Predicting")
        predictions = self.model.predict(input_data, batch_size=1)
        predictions_rescaled = self._rescale_data(predictions)
        targets_rescaled = self._rescale_data(targets)
        return predictions_rescaled, targets_rescaled

    def _copy_trained_weights_to_model_with_different_batch_size(self) -> None:
        trained_lstm_weights = self.model.lstm.get_weights()
        params = self.hyper_parameters["lstm"].copy()
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
            lambda trial: local_univariate_cnn_ae_lstm_keras_objective(
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
        save_path = f"{path}{self.get_name()}.h5"
        # self.model.save_weights(save_path)
        return save_path

    def load(self, path: str) -> None:
        load_path = f"{path}{self.get_name()}.h5"
        # self.model.load_weights(load_path)
