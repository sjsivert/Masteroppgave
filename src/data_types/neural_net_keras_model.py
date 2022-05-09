import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import tensorflow as tf
import optuna
from matplotlib.figure import Figure

from sklearn.preprocessing import StandardScaler
from optuna import Study
from optuna.trial import FrozenTrial
from optuna.visualization import (
    plot_edf,
    plot_intermediate_values,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_slice,
)
from src.utils.keras_error_calculations import (
    config_metrics_to_keras_metrics,
    generate_error_metrics_dict,
    keras_mase,
    keras_mase_periodic,
    keras_smape,
)
from numpy import ndarray
from src.data_types.i_model import IModel
from src.pipelines import local_univariate_lstm_keras_pipeline as lstm_keras_pipeline
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from src.save_experiment_source.neptune_save_source import NeptuneSaveSource
from src.utils.visuals import visualize_data_series, visualize_data_series_with_specified_x_axis
import torch


class NeuralNetKerasModel(IModel, ABC):
    def __init__(
        self,
        log_sources: List[ILogTrainingSource],
        time_series_id: str,
        params: Dict,
        optuna_trial: Optional[optuna.trial.Trial] = None,
        # pipeline: Any = lstm_keras_pipeline.local_univariate_lstm_keras_pipeline
        pipeline: Any = multivariate_pipeline.local_multivariate_lstm_keras_pipeline,
    ):
        # Init global variables
        self.model = None
        self.figures: List[Figure] = []
        self.metrics: Dict = {}
        self.log_sources: List[ILogTrainingSource] = log_sources
        self.name = time_series_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = params["batch_size"]
        # TODO: What is training size?
        self.input_window_size = params["input_window_size"]
        self.output_window_size = params["output_window_size"]
        self.pipeline = pipeline

        self.training_data_loader = None
        self.validation_data_loader = None
        self.testing_data_loader = None
        self.min_max_scaler = None
        self.optuna_trial = optuna_trial
        self.hyper_parameters = params.copy()

        logging.info("Running model on device: {}".format(self.device))

        self.init_neural_network(params)
        self.should_shuffle_batches = params["should_shuffle_batches"]
        # Defining data set varaibles
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None

    @abstractmethod
    def init_neural_network(self, params: dict, logger=None, **xargs) -> None:
        return NotImplemented

    def get_name(self) -> str:
        return self.name

    def get_model(self):
        return self.model

    def process_data(self, data_set: Any, training_size: float) -> None:
        data_pipeline = self.pipeline(
            data_set=data_set,
            cat_id=self.get_name(),
            input_window_size=self.input_window_size,
            output_window_size=self.output_window_size,
        )
        logging.info(f"Data Pipeline for {self.get_name()}: {data_pipeline}")
        for log_source in self.log_sources:
            log_source.log_pipeline_steps(data_pipeline.__repr__())

        (
            training_data,
            testing_data,
            self.min_max_scaler,
            self.training_data_no_windows,
            self.training_data_without_diff,
        ) = data_pipeline.run()
        training_data, validation_data, testing_data = self.split_data_sets(
            training_data=training_data, testing_data=testing_data
        )
        self.x_train, self.y_train = training_data
        self.x_val, self.y_val = validation_data
        self.x_test, self.y_test = testing_data

    def split_data_sets(self, training_data, testing_data):
        # Do not look a the code in the next 11 lines below, it is ugly and I am not proud of it
        examples_to_drop_to_make_all_batches_same_size = training_data[0].shape[0] % self.batch_size

        examples_to_drop_to_make_all_batches_same_size = (
            -self.hyper_parameters["output_window_size"]
            if examples_to_drop_to_make_all_batches_same_size == 0 and self.batch_size == 1
            else -examples_to_drop_to_make_all_batches_same_size
        )
        examples_to_drop_to_make_all_batches_same_size = (
            None
            if examples_to_drop_to_make_all_batches_same_size == 0
            else examples_to_drop_to_make_all_batches_same_size
        )

        logging.info(
            f"Examples to drop to make all batches same size: {examples_to_drop_to_make_all_batches_same_size}"
        )
        x_train, y_train = (
            training_data[0][:examples_to_drop_to_make_all_batches_same_size],
            training_data[1][:examples_to_drop_to_make_all_batches_same_size],
        )
        x_val, y_val = (
            x_train[-self.batch_size :],
            y_train[-self.batch_size :],
        )
        x_train = x_train[: -self.batch_size]
        y_train = y_train[: -self.batch_size]
        x_test, y_test = testing_data[0], testing_data[1]
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)

    def log_trial(self, study: Study, trial: FrozenTrial) -> None:
        for log_source in self.log_sources:
            trial_info = trial.params
            trial_info["score"] = trial.value
            trial_info["Trial number"] = trial.number
            log_source.log_tuning_metrics(
                {"" + str(self.get_name()) + ": " + "trial.number": {"Parameters": trial_info}}
            )

    # Optuna plot
    def _generate_optuna_plots(self, study: Study) -> None:
        # TODO: Currently getting error Figure has not attribute axes. Fix
        self.figures.append(
            plot_slice(study).update_layout(title=f"{self.get_name()} - Plot Slice")
        )
        self.figures.append(plot_edf(study).update_layout(title=f"{self.get_name()} - Plot EDF"))
        self.figures.append(
            plot_intermediate_values(study).update_layout(
                title=f"{self.get_name()} - Plot Intermediate Values"
            )
        )
        self.figures.append(
            plot_optimization_history(study).update_layout(
                title=f"{self.get_name()} - Plot Optimization History"
            )
        )
        self.figures.append(
            plot_parallel_coordinate(study).update_layout(
                title=f"{self.get_name()} - Plot Parallel Coordinate"
            )
        )

    # Shared methods for retrieving information
    def get_figures(self) -> List[Figure]:
        """
        Return a list of figures created by the model for visualization
        """
        return self.figures

    def get_metrics(self) -> Dict:
        """
        Fetch metrics from model training or testing
        """
        return self.metrics

    def save(self, path: str) -> str:
        """
        Save the model to the specified path.
        :returns: Path to saved model file
        """
        raise NotImplementedError()

    def load(self, path: str) -> IModel:
        """
        Load the model from the specified path.
        The correct model should already be created as self.model.
        The load function only sets the pretrained verctor values from the saved model.
        """
        raise NotImplementedError()

    def get_predictions(self) -> Optional[Dict]:
        """
        Returns the predicted values if test() has been called.
        """
        raise NotImplementedError()

    def _rescale_data(self, data: ndarray, scaler=None) -> ndarray:
        scaler = self.min_max_scaler if scaler is None else scaler
        return scaler.inverse_transform(data) if scaler is not None else data

    def _reverse_pipeline(self, predictions, min_max_scaler, original_data):
        predictions_scaled = self._rescale_data(predictions, min_max_scaler)
        # predictions_reversed_diff = reverse_differencing_forecast(
        #     noise=predictions_scaled, last_observed=original_data[-1]
        # )
        predictions_reversed_diff = predictions_scaled
        # predictions_added_variance = reverse_decrease_variance(predictions_reversed_diff)
        predictions_added_variance = predictions_reversed_diff

        return predictions_added_variance

    def _reverse_pipeline_training(self, training_data, original_data, scaler: StandardScaler = None):
        # visualize_data_series(
        #     title=f"original data",
        #     data_series=[training_data],
        #     data_labels=["Targets", ],
        #     colors=["blue",],
        #     x_label="Time",
        #     y_label="Interest",
        # ).savefig("original_training_data_before_post_prosessing.png")

        rescaled = self._rescale_data(training_data.reshape(1, -1), scaler)
        # visualize_data_series(
        #     title=f"rescaled data",
        #     data_series=[rescaled],
        #     data_labels=["Targets", ],
        #     colors=["blue",],
        #     x_label="Time",
        #     y_label="Interest",
        # ).savefig("rescaled_data.png")

        # reverse_diff = reverse_differencing(rescaled, original_data)
        reverse_diff = rescaled

        # visualize_data_series(
        #     title=f"re-reverse_diff",
        #     data_series=[reverse_diff],
        #     data_labels=["Targets", ],
        #     colors=["blue",],
        #     x_label="Time",
        #     y_label="Interest",
        # ).savefig("re-reverse-diff.png")
        # increase_variance = reverse_decrease_variance(reverse_diff)
        increase_variance = reverse_diff
        # visualize_data_series(
        #     title=f"incverage_variance",
        #     data_series=[increase_variance],
        #     data_labels=["Targets", ],
        #     colors=["blue",],
        #     x_label="Time",
        #     y_label="Interest",
        # ).savefig("increase_variances.png")
        return increase_variance

    def predict_and_rescale(self, input_data: ndarray, targets: ndarray, model: IModel) -> ndarray:
        logging.info("Predicting")
        predictions = model.predict(input_data, batch_size=1)
        predictions_rescaled = self._rescale_data(predictions)
        targets_rescaled = self._rescale_data(targets)
        # predictions_rescaled = self._rescale_data(DataFrame(predictions))
        # targets_rescaled = self._rescale_data(DataFrame(predictions))

        # After fixing multivariate pipeline there was a bug that made rescaling not work
        # Therefore this is disabled for now
        # predictions_rescaled = predictions
        # targets_rescaled = targets

        return predictions_rescaled, targets_rescaled

    def custom_evaluate(self, x_test, y_test, model, scaler: StandardScaler = None):
        scaler = self.min_max_scaler if scaler is None else scaler
        predictions = model.predict(x_test, batch_size=1)

        predictions_re_composed = self._reverse_pipeline(
            predictions, scaler, self.training_data_without_diff
        )
        # TODO Post processing
        results = {}
        kerast_metrics_to_calculate = [
            tf.keras.metrics.MeanSquaredError,
            tf.keras.metrics.MeanAbsoluteError,
            tf.keras.metrics.MeanAbsolutePercentageError,
        ]

        for metric_func in kerast_metrics_to_calculate:
            metric = metric_func()
            metric.update_state(y_test, predictions_re_composed)
            results[metric.name] = metric.result().numpy()
        results["mase"] = keras_mase(y_true=y_test, y_pred=predictions_re_composed).numpy()
        results["smape"] = keras_smape(y_true=y_test, y_pred=predictions_re_composed)
        return results, predictions

    # LSTM methods training while not tuning, visualization of predictions
    def _no_tuning_visualization_predictions(
        self, training_predictions, validation_predictions, history
    ):
        training_data_original_scale = self._reverse_pipeline_training(
            self.training_data_no_windows, self.training_data_without_diff
        )
        training_predictions_original_scale = self._reverse_pipeline_training(
            training_predictions[:, 0], self.training_data_without_diff
        )
        self._visualize_predictions(
            (training_data_original_scale.flatten()),
            (training_predictions_original_scale.flatten()),
            "Training predictions original scale",
        )
        self._visualize_predictions(
            (self.training_data_no_windows.flatten()),
            (training_predictions[:, 0].flatten()),
            "Training predictions",
        )
        self._visualize_predictions(
            (self.x_train[:, 0, 0].flatten()),
            (self.y_train[0, 0].flatten()),
            "Training predictions without validation set",
        )

        self._visualize_predictions(
            self.y_val.flatten(),
            validation_predictions.flatten()
            if validation_predictions.shape[0] > 1
            else validation_predictions[:, 0].flatten(),
            "Validation predictions",
        )
        self._visualize_errors(
            [history["loss"], history["val_loss"]], ["Training_errors", "Validation_errors"]
        )

    def predict_and_reverse_pipeline(
        self,
        input_data: ndarray,
        targets: ndarray,
        min_max_scaler: StandardScaler,
        original_data,
        model,
    ) -> ndarray:
        # reverse training data
        logging.info("Predicting")
        predictions = model.predict(input_data, batch_size=1)
        predictions_reversed_pipeline = self._reverse_pipeline(
            predictions, min_max_scaler, original_data
        )
        return predictions_reversed_pipeline, targets

    def _lstm_test_scale_predictions(
        self, x_train, y_train, x_test, y_test, test_metrics, test_predictions, model
    ):
        test_predictions_reversed, test_targets_reversted = self.predict_and_reverse_pipeline(
            self.x_test, self.y_test, self.min_max_scaler, self.training_data_without_diff, model
        )
        # last_period_targets = (
        #     self.min_max_scaler.inverse_transform(x_test[:, -self.output_window_size:])
        #     if self.min_max_scaler
        #     else x_test[:, -self.output_window_size:]
        # )
        y_true_last_period = self.x_test[:, -self.output_window_size :, 0]
        last_period_targets = self._reverse_pipeline_training(
            training_data=y_true_last_period,
            original_data=self.training_data_without_diff[-self.output_window_size :, :],
        )

        mase_periode, y_true_last_period = (
            keras_mase_periodic(
                y_true=self.y_test,
                y_true_last_period=last_period_targets,
                y_pred=test_predictions_reversed,
            )
            if self.min_max_scaler is not None
            else (420, None)
        )

        # Custom evaluate function with rescale before metrics
        model.reset_states()
        model.predict(x_train, batch_size=1)
        custom_metrics, _ = self.custom_evaluate(
            x_test=x_test, y_test=y_test, model=model, scaler=self.min_max_scaler
        )
        self.metrics.update(custom_metrics)
        logging.info("CUSTOM METRRICS-----------\n", custom_metrics)
        # self.metrics.update(custom_metrics)
        print("Mase", mase_periode)
        test_metrics[f"test_MASE_{self.output_window_size}_DAYS"] = mase_periode.numpy()
        self.metrics.update(custom_metrics)

        # Visualize
        self._visualize_test_predictions(
            test_predictions_reversed, test_predictions, last_period_targets, x_test
        )

        self.metrics.update(test_metrics)
        training_data_original_scale = self._reverse_pipeline_training(
            self.training_data_no_windows, self.training_data_without_diff
        )

    def _visualize_test_predictions(
        self, test_predictions_reversed, test_predictions, last_period_targets, x_test
    ):
        self._visualize_predictions(
            (self.y_test.flatten()),
            (test_predictions_reversed.flatten()),
            "Test predictions rescaled",
        )
        self._visualize_predictions(
            (self.y_test.flatten()),
            (test_predictions.flatten()),
            "Test predictions not scaled",
        )
        self._visualize_predictions_and_last_period(
            (self.y_test.flatten()),
            (test_predictions_reversed.flatten()),
            last_period_targets.flatten(),
            "Test predictions with last period targets",
        )
        x_test_values = self._reverse_pipeline_training(
            training_data=x_test[:, -self.output_window_size :],
            original_data=self.training_data_without_diff[-self.output_window_size :],
        )
        self._visualize_predictions_with_context(
            context=x_test_values.flatten(),
            targets=self.y_test.flatten(),
            predictions=test_predictions_reversed.flatten(),
        )

    # Visualization
    def _visualize_errors(
        self, errors: List[List[float]], labels: List[str] = ["Training error", "Validation error"]
    ) -> None:
        # Visualize training and validation loss
        self.figures.append(
            visualize_data_series(
                title=f"{self.get_name()}# Training and Validation error",
                data_series=[x for x in errors],
                data_labels=labels,
                colors=["blue", "orange", "red", "green"],
                x_label="Epoch",
                y_label="Error",
            )
        )

    def _visualize_predictions(self, targets, predictions, name: str):
        self.figures.append(
            visualize_data_series(
                title=f"{self.get_name()}# {name}",
                data_series=[targets, predictions],
                data_labels=["Targets", "Predictions"],
                colors=["blue", "orange"],
                x_label="Time",
                y_label="Interest",
            )
        )

    def _visualize_predictions_and_last_period(self, targets, predictions, last_period, name: str):
        self.figures.append(
            visualize_data_series(
                title=f"{self.get_name()}# {name}",
                data_series=[targets, last_period, predictions],
                data_labels=["Targets", "last_period", "Predictions"],
                colors=["blue", "green", "orange"],
                x_label="Time",
                y_label="Interest",
            )
        )

    def _visualize_predictions_with_context(self, context, targets, predictions):
        context_axis = [x for x in range(len(context))]
        predictions_axis = [x + len(context) for x in range(len(predictions))]
        self.figures.append(
            visualize_data_series_with_specified_x_axis(
                title=f"{self.get_name()}# Test predictions with context",
                data_series=[context, targets, predictions],
                data_axis=[context_axis, predictions_axis, predictions_axis],
                data_labels=["Contextual Data", "Targets", "Prediction"],
                colors=["blue", "green", "orange"],
                x_label="date",
                y_label="Interest",
            )
        )
