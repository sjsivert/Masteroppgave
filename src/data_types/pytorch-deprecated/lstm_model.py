from typing import Any, List, Optional

import optuna
import pytorch_lightning as pl
from src.data_types.modules.lstm_lightning_module import LSTMLightning
from src.data_types.neural_net_model import NeuralNetModel
from src.optuna_tuning.loca_univariate_lstm_objective import local_univariate_lstm_objective
from src.pipelines import local_univariate_lstm_pipeline as lstm_pipeline
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from src.save_experiment_source.local_checkpoint_save_source import LocalCheckpointSaveSource
from src.utils.pytorch_error_calculations import *


class LstmModel(NeuralNetModel):
    def __init__(
        self,
        log_sources: List[ILogTrainingSource],
        time_series_id: str,
        params: Dict,
        optuna_trial: Optional[optuna.trial.Trial] = None,
    ):
        super(LstmModel, self).__init__(
            log_sources,
            time_series_id,
            params,
            optuna_trial,
            pipeline=lstm_pipeline.local_univariate_lstm_pipeline,
        )

    def init_neural_network(self, params: dict, logger=None, **xargs) -> None:
        # Creating LSTM module
        self.model = LSTMLightning(**params)
        self.trainer = pl.Trainer(
            enable_checkpointing=False,
            max_epochs=params["number_of_epochs"],
            deterministic=True,
            logger=self._get_neptune_run_from_save_sources() if logger is None else logger,
            auto_select_gpus=True if self.device == "cuda" else False,
            # gpus=1 if torch.cuda.is_available() else 0,
            # gpus= [i for i in range(torch.cuda.device_count())],
            gpus=-1 if torch.cuda.is_available() else 0,
            **xargs,
        )

    def train(self, epochs: int = None, **xargs) -> Dict:
        # Visualization

        self.trainer.fit(
            self.model,
            train_dataloaders=self.training_data_loader,
            val_dataloaders=self.validation_data_loader,
        )
        training_targets, training_predictions = self.model.visualize_predictions(
            self.training_data_loader
        )

        self.metrics["training_error"] = self.model.training_errors[-1]
        self.metrics["validation_error"] = self.model.validation_errors[-1]
        self._visualize_predictions(training_targets, training_predictions, "Training predictions")
        self._visualize_errors(
            self.model.training_errors,
            self.model.validation_errors,
        )
        return self.metrics

    def test(self, predictive_period: int = 6, single_step: bool = False) -> Dict:
        self.trainer.test(self.model, dataloaders=self.testing_data_loader)
        # Visualize predictions -> TODO: Add multi step visualization
        test_targets, test_predictions = self.model.visualize_predictions(self.testing_data_loader)
        self._visualize_predictions(test_targets, test_predictions, "Test predictions")
        # Trainer get list of errors
        logging.info(f"Testing error: {self.model.test_losses_dict}.")
        self.metrics.update(self.model.test_losses_dict)
        return self.model.test_losses_dict

    def method_evaluation(
        self,
        parameters: Any,
        metric: str,
        singe_step: bool = True,
    ) -> Dict[str, Dict[str, str]]:
        """
        Evaluate model and calculate error of predictions for used for tuning evaluation
        """
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
