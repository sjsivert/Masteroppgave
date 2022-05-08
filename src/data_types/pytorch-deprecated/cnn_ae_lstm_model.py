import logging
from typing import List, Dict, Optional, Any

import torch
from src.data_types.modules.cnn_ae_lstm_module import CNN_AE_LSTM
from src.data_types.modules.cnn_ae_module import CNN_AE
from src.data_types.modules.lstm_lightning_module import LSTMLightning
from src.data_types.neural_net_model import NeuralNetModel
from src.save_experiment_source.i_log_training_source import ILogTrainingSource

import optuna
import pytorch_lightning as pl
from src.pipelines import local_univariate_lstm_pipeline as lstm_pipeline


class CNNAELSTMModel(NeuralNetModel):
    def __init__(
        self,
        log_sources: List[ILogTrainingSource],
        time_series_id: str,
        params: Dict,
        optuna_trial: Optional[optuna.trial.Trial] = None,
    ):
        super(CNNAELSTMModel, self).__init__(
            log_sources,
            time_series_id,
            params,
            optuna_trial,
            pipeline=lstm_pipeline.local_univariate_lstm_pipeline,
        )

    def init_neural_network(self, params: dict, logger=None, **xargs) -> None:
        self.ae = CNN_AE()
        self.lstm = LSTMLightning(**params)
        self.model = CNN_AE_LSTM(autoencoder=self.ae, lstm=self.lstm)
        self.ae_trainer = pl.Trainer(
            enable_checkpointing=False,
            max_epochs=50,
            deterministic=True,
            logger=self._get_neptune_run_from_save_sources() if logger is None else logger,
            auto_select_gpus=True if self.device == "cuda" else False,
            gpus=1 if torch.cuda.is_available() else 0,
            **xargs,
        )
        self.trainer = pl.Trainer(
            enable_checkpointing=False,
            max_epochs=300,
            deterministic=True,
            logger=self._get_neptune_run_from_save_sources() if logger is None else logger,
            auto_select_gpus=True if self.device == "cuda" else False,
            gpus=1 if torch.cuda.is_available() else 0,
            **xargs,
        )

    def train(self, epochs: int = None, **xargs) -> Dict:
        logging.info("Training")
        # Training the Auto encoder
        self.ae_trainer.fit(
            self.ae,
            train_dataloader=self.training_data_loader,
            val_dataloaders=self.validation_data_loader,
        )
        # Visualize Autoencoder results on test and validation data
        training_targets, training_predictions = self.ae.visualize_predictions(
            self.training_data_loader,
        )
        self._visualize_predictions(
            training_targets, training_predictions, "Auto encoder training set"
        )
        validation_targets, validation_predictions = self.ae.visualize_predictions(
            self.validation_data_loader, first=False
        )
        self._visualize_predictions(
            validation_targets, validation_predictions, "Auto encoder validation set"
        )

        # Training the AE and LSTM model
        self.trainer.fit(
            self.model,
            train_dataloader=self.training_data_loader,
            val_dataloaders=self.validation_data_loader,
        )
        training_targets, training_predictions = self.model.visualize_predictions(
            self.training_data_loader
        )
        self._visualize_predictions(
            training_targets, training_predictions, "CNN-AE-LSTM training set"
        )
        validation_targets, validation_predictions = self.model.visualize_predictions(
            self.validation_data_loader
        )
        self._visualize_predictions(
            validation_targets, validation_predictions, "CNN-AE-LSTM validation set"
        )
        self._visualize_errors(self.model.training_error, self.model.validation_error)

        self.metrics["training_error"] = self.model.training_error[-1]
        self.metrics["validation_error"] = self.model.validation_error[-1]
        return self.metrics

    def test(self, predictive_period: int = 7, single_step: bool = False) -> Dict:
        logging.info("Testing CNN-AE model")
        # Test AE model
        self.ae_trainer.test(self.ae, self.testing_data_loader)
        test_targets, test_predictions = self.ae.visualize_predictions(self.testing_data_loader)
        self._visualize_predictions(test_targets, test_predictions, "Autoencoder test predictions")
        # Testing AE-LSTM model
        self.trainer.test(self.model, self.testing_data_loader)
        test_targets, test_predictions = self.model.visualize_predictions(self.testing_data_loader)
        self._visualize_predictions(
            test_targets, test_predictions, "CNN-AE and LSTM test predictions"
        )
        self.metrics["test error"] = self.model.test_error
        return self.metrics

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
