import logging
import math
import multiprocessing as mp
import sys
import warnings
from abc import ABC
from collections import OrderedDict
from typing import Dict, List, Tuple

import pandas as pd
import pmdarima as pm
from genpipes.compose import Pipeline
from matplotlib.figure import Figure
from pandas import DataFrame, Series
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data_types.i_model import IModel
from src.pipelines.arima_model_pipeline import arima_model_pipeline
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from src.utils.error_calculations import calculate_error, calculate_periodic_mase
from src.utils.visuals import visualize_data_series
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

# from pathos.multiprocessing import ProcessingPool as Pool


class ArimaModel(IModel, ABC):

    # TODO! ARIMA model fails on Linux in cases where the model is interpreted as non-stationary.
    # TODO: It should not be an issue on Windows, but further validation of the problem is needed.
    # TODO: The error is recreatable with the ARIMA config (5,4,5)
    # TODO! Add try catch to the training of the ARIMA model for instances like these

    # TODO! Set random SEED
    def __init__(
        self,
        log_sources: List[ILogTrainingSource],
        name: str = "placeholder",
        hyperparameters: Dict[str, int] = OrderedDict(
            {"p": 2, "d": 1, "q": 3, "P": 3, "D": 2, "Q": 1, "s": 12}
        ),
        seasonal: bool = True,
    ):
        self.data_pipeline = None
        self.value_approximation = None
        self.training_data = None
        self.validation_data = None
        self.test_data = None

        self.log_sources: List[ILogTrainingSource] = log_sources
        # Model name
        self.name: str = name
        # The ARIMA model must be instanced with data, thus this is done during training
        self.model = None
        self.order = list(hyperparameters.values())  # Tuple defining the ARIMA order variable
        self.predictions: DataFrame = None
        # Visualization
        self.figures: List[Figure] = []
        self.metrics: Dict = {}

        self.training_periode = (
            0  # Integer defining the number of data-points used to train the ARIMA model
        )
        self.training_residuals = None  # Dataframe of training residuals
        self.seasonal = seasonal
        super().__init__()

    def process_data(
        self, data_set: DataFrame, training_size: float = 7
    ) -> Tuple[DataFrame, DataFrame]:
        self.data_pipeline = arima_model_pipeline(
            data_set=data_set, cat_id=self.get_name(), forecast_size=int(training_size)
        )
        logging.info(
            f"ArimaModel: Data pipeline created for {self.get_name()}\n {self.data_pipeline.__repr__()}"
        )
        for log_source in self.log_sources:
            log_source.log_pipeline_steps(self.data_pipeline.__repr__())

        self.training_data, self.validation_data, self.test_data, _ = self.data_pipeline.run()
        return self.training_data, self.test_data

    def get_name(self) -> str:
        return self.name

    def train(self, epochs: int = 10) -> Dict:
        if self.training_data is None or self.test_data is None:
            raise ValueError(
                "Model does not have test og training data. Please call process_data() first."
            )

        train_val_data = pd.concat([self.training_data, self.validation_data])
        self.training_periode = len(train_val_data)
        if self.seasonal:  # SARIMA
            arima_model = ARIMA(train_val_data, order=self.order[:3], seasonal_order=self.order[3:])
        else:  # ARIMA
            arima_model = ARIMA(train_val_data, order=self.order[:3])
        arima_model_res = arima_model.fit()
        logging.info(arima_model_res.summary())

        self.model = arima_model_res
        self.value_approximation = DataFrame(self.model.predict(0, self.training_periode - 1))
        # Figures
        self._visualize_training(train_val_data, self.value_approximation)
        # Metrics
        metrics = calculate_error(
            train_val_data["interest"], self.value_approximation["predicted_mean"]
        )

        self.metrics = dict(map(lambda x: (f"Training_{x[0]}", x[1]), metrics.items()))
        logging.info(f"Training metrics: {self.metrics}")
        for log_source in self.log_sources:
            log_source.log_metrics({self.name: self.metrics}, 0)

        return metrics

    def test(self, predictive_period: int = 5, single_step: bool = False) -> Dict:
        if self.training_data is None or self.test_data is None:
            raise ValueError(
                "Model does not have test og training data. Please call process_data() first."
            )
        if_predictive_period_longer_than_dataset_use_max_length = (
            predictive_period if predictive_period <= len(self.test_data) else len(self.test_data)
        )
        predictive_period = if_predictive_period_longer_than_dataset_use_max_length
        if single_step:
            logging.info("ArimaModel: Single step prediction")
            value_predictions = ArimaModel._single_step_prediction(
                model=self.model, test_set=self.test_data
            )
            self.predictions = value_predictions[0][:predictive_period]
        else:
            # Test data
            logging.info("ArimaModel: Multi step prediction")
            value_predictions = self.model.forecast(predictive_period)
            self.predictions = value_predictions

        # Figures
        train_val_data = pd.concat([self.training_data, self.validation_data])
        self._visualize_testing(
            train_val_data, self.test_data[:predictive_period], self.predictions
        )
        # Metrics
        metrics = calculate_error(self.test_data["interest"][:predictive_period], self.predictions)
        self.metrics = dict(map(lambda x: (f"Testing_{x[0]}", x[1]), metrics.items()))
        # Periodic naive mase
        periodic_mase = calculate_periodic_mase(
            self.test_data["interest"][:predictive_period],
            self.predictions,
            train_val_data["interest"][-predictive_period:],
        )
        self.metrics[f"Testing_MASE_{str(predictive_period)}_days"] = periodic_mase

        logging.info(
            f"\nPredictions ahead: {predictive_period}\n"
            + f"Predicting from {self.training_periode} to {self.training_periode + predictive_period - 1}\n"
            + f"Testing metrics: {self.metrics}"
        )

        return metrics

    def get_metrics(self) -> Dict:
        return self.metrics

    def get_figures(self) -> List[Figure]:
        return self.figures

    def get_predictions(self) -> DataFrame:
        return self.predictions

    def save(self, path: str) -> str:
        save_path = f"{path}Arima_{self.get_name()}.pkl"
        if self.model is not None:
            self.model.save(save_path)
        return save_path

    def load(self, path: str) -> IModel:
        try:
            load_path = f"{path}Arima_{self.get_name()}.pkl"
            loaded_model = ARIMAResults.load(load_path)
            self.model = loaded_model
            return loaded_model
        except FileNotFoundError:
            raise FileNotFoundError(f"The stored Arima model is not found at path: {load_path}")

    def _visualize_training(self, training, approximation):
        # Only training data
        self.figures.append(
            visualize_data_series(
                title=f"{self.get_name()}# Training data",
                data_series=[training],
                data_labels=["Training data"],
                colors=["blue"],
                x_label="date",
                y_label="Interest",
            )
        )
        # Training and approx data
        self.figures.append(
            visualize_data_series(
                title=f"{self.get_name()}# Training data approximation",
                data_series=[training, approximation],
                data_labels=["True value", "Approximation"],
                colors=["blue", "orange"],
                x_label="date",
                y_label="Interest",
            )
        )

    def _visualize_testing(self, training_set, testing_set, prediction_set):
        # Testing data and prediction
        self.figures.append(
            visualize_data_series(
                title=f"{self.get_name()}# Data Prediction",
                data_series=[testing_set, prediction_set],
                data_labels=["True value", "Prediction"],
                colors=["blue", "red"],
                x_label="date",
                y_label="Interest",
            )
        )
        # True training data, test data, and predictions
        self.figures.append(
            visualize_data_series(
                title=f"{self.get_name()}# Predictions",
                data_series=[training_set, testing_set, prediction_set],
                data_labels=["Test data", "Testing data", "Prediction"],
                colors=["blue", "orange", "red"],
                x_label="date",
                y_label="Interest",
            )
        )
        # True training data, test data, and predictions
        self.figures.append(
            visualize_data_series(
                title=f"{self.get_name()}# Predictions with context",
                data_series=[training_set[-17:], testing_set, prediction_set],
                data_labels=["Test data", "Testing data", "Prediction"],
                colors=["blue", "orange", "red"],
                x_label="date",
                y_label="Interest",
            )
        )

    @staticmethod
    def _single_step_prediction(model: ARIMAResults, test_set: DataFrame) -> DataFrame:
        forecast = []
        for index, row in test_set.iterrows():
            data_point = test_set.loc[index:index]
            prediction = Series(model.forecast(1))
            forecast.append(
                prediction,  # Single step forecast
            )
            model = model.extend(data_point)
        forecast = DataFrame(pd.concat(DataFrame(x) for x in forecast))
        return forecast

    def auto_arima(
        self,
        p=0,
        d=1,
        q=0,
        max_p=1,
        max_d=1,
        max_q=1,
        P=0,
        D=1,
        Q=0,
        max_P=1,
        max_D=1,
        max_Q=1,
        m=4,
        epochs=50,
        metric="mae",
    ):
        train_val_data = pd.concat([self.training_data, self.validation_data])
        model = pm.auto_arima(
            train_val_data,
            start_p=p,
            d=d,
            start_q=q,
            max_p=max_p,
            max_q=max_q,
            max_d=max_d,
            start_P=P,
            D=D,
            start_Q=Q,
            max_P=max_P,
            max_D=max_D,
            max_Q=max_Q,
            m=m,
            seasonal=self.seasonal,
            error_action="warn",
            trace=True,
            suppress_warnings=True,
            stepwise=True,
            random_state=40,
            n_fits=epochs,
            scoring=metric,
            information_criterion="bic",  # Bayesian Information Criterion
        )
        logging.info(model.summary())
        selected_order = f"{model.order}{model.seasonal_order}"
        forecast = model.predict(len(self.test_data))
        forecast = DataFrame(forecast, index=self.test_data.index)
        metrics = calculate_error(self.test_data, forecast)
        self.metrics = dict(map(lambda x: (f"Testing_{x[0]}", x[1]), metrics.items()))
        self._visualize_testing(train_val_data, self.test_data, forecast)
        logging.info(self.metrics)
        return {selected_order: metrics}

    # Static method evaluating an Arima model
    def method_evaluation(
        self,
        parameters: List,
        metric: str,
        single_step: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        assert self.training_data is not None, "Training data is not loaded"
        assert self.validation_data is not None, "Validation data is not loaded"

        error_param_set = {}
        # Try catch block for numpy LU decomposition error
        cores = mp.cpu_count()
        pool = mp.Pool(cores)
        results = []
        for order in parameters:
            result = pool.apply_async(
                ArimaModel._eval_arima,
                args=(order, self.training_data, self.validation_data, single_step, self.seasonal),
            )
            results.append(result)
        pool.close()
        pool.join()
        # Complete waiting for async operations
        [result.wait() for result in results]
        for result in results:
            if result._value[0] is not None:
                error_param_set[result._value[0]] = result._value[1]
        # Sort the values to ease the evaluation of model parameters
        error_param_set = dict(
            sorted(
                error_param_set.items(),
                key=lambda x: float("inf") if math.isnan(x[1][metric]) else x[1][metric],
                reverse=False,
            )
        )
        return error_param_set

    @staticmethod
    def _eval_arima(
        order: Tuple[Tuple[int, int, int], Tuple[int, int, int, int]],
        training_set: DataFrame,
        test_set: DataFrame,
        single_step: bool,
        seasonal: bool = True,
    ) -> Tuple[str, float]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # Create and fit model with training data. Either ARIMA or SARIMA
                if seasonal:
                    arima_base = ARIMA(training_set, order=order[0], seasonal_order=order[1])
                else:
                    arima_base = ARIMA(training_set, order=order[0])
                model = arima_base.fit()
                if not single_step:
                    forecast = model.forecast(len(test_set))
                    forecast = DataFrame(forecast)
                # Evaluate model with single step evaluation and Walk-forward validation
                else:
                    forecast = ArimaModel._single_step_prediction(model=model, test_set=test_set)
                error = calculate_error(test_set, forecast)
                logging.info(f"Tuning ARIMA model. Parameters {order} used.")
                return f"{order}", error
            except KeyboardInterrupt:
                sys.exit()
                return None, None
            except Exception as e:
                logging.info(e)
                logging.info(
                    f"Tuning ARIMA model {order} got an error. Calculations not completed."
                )
            return None, None

    def get_data_pipeline(self) -> Pipeline:
        return self.data_pipeline

    def __repr__(self):
        return f"<ArimaModel prouct_id: {self.get_name()}>"
