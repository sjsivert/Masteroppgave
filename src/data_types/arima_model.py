import logging
from typing import Dict, List, Tuple

import pandas as pd
from matplotlib.figure import Figure
from pandas import DataFrame, Series
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data_types.i_model import IModel
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from src.utils.error_calculations import calculate_error, calculate_mse
from src.utils.visuals import visualize_data_series
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults


class ArimaModel(IModel):

    # TODO! ARIMA model fails on Linux in cases where the model is interpreted as non-stationary.
    # TODO: It should not be an issue on Windows, but further validation of the problem is needed.
    # TODO: The error is recreatable with the ARIMA config (5,4,5)
    # TODO! Add try catch to the training of the ARIMA model for instances like these

    # TODO! Set random SEED
    def __init__(
        self,
        log_sources: List[ILogTrainingSource],
        name: str = "placeholder",
        order: Tuple[int, int, int] = (0, 0, 0),
    ):
        super().__init__(log_sources=log_sources, name=name)
        # The ARIMA model must be instanced with data, thus this is done during training
        self.model = None
        self.order = order  # Tuple defining the ARIMA order variable
        self.training_periode = (
            0  # Integer defining the number of data-points used to train the ARIMA model
        )
        self.training_residuals = None  # Dataframe of training residuals

    def train(self, data_set: DataFrame, epochs: int = 10) -> Dict:
        # TODO: Fix training size
        self.training_periode = len(data_set)
        arima_model = ARIMA(data_set, order=self.order)
        arima_model_res = arima_model.fit()

        logging.info(arima_model_res.summary())

        self.model = arima_model_res
        self.value_approximation = self.model.predict(0, self.training_periode - 1)
        # Figures
        self._visualize_training(data_set, self.value_approximation)
        # Metrics
        metrics = calculate_error(data_set, self.value_approximation)
        self.metrics = dict(map(lambda x: (f"Training_{x[0]}", x[1]), metrics.items()))
        logging.info(f"Training metrics: {self.metrics}")

        return metrics

    def test(self, test_data_set: DataFrame, predictive_period: int = 5) -> Dict:
        if_predictive_period_longer_than_dataset_use_max_length = (
            predictive_period if predictive_period <= len(test_data_set) else len(test_data_set)
        )
        predictive_period = if_predictive_period_longer_than_dataset_use_max_length

        value_predictions = self.model.predict(
            self.training_periode, self.training_periode + predictive_period - 1
        )
        self.predictions = DataFrame(value_predictions)
        # Figures
        self._visualize_testing(test_data_set[:predictive_period], self.predictions)
        # Metrics
        metrics = calculate_error(test_data_set[:predictive_period], self.predictions)
        self.metrics = dict(map(lambda x: (f"Testing_{x[0]}", x[1]), metrics.items()))

        logging.info(
            f"\nPredictions ahead: {predictive_period}\n"
            + f"Predicting from {self.training_periode} to {self.training_periode + predictive_period - 1}\n"
            + f"Predictions:\n{value_predictions}\n"
            + f"Testing metrics: {self.metrics}"
        )

        return metrics

    def get_metrics(self) -> Dict:
        return self.metrics

    def get_figures(self) -> List[Figure]:
        return self.figures

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
                title="Training data",
                data_series=[training],
                data_labels=["Training data"],
                colors=["blue"],
                x_label="date",
                y_label="value",
            )
        )
        # Training and approx data
        self.figures.append(
            visualize_data_series(
                title="Training data approximation",
                data_series=[training, approximation],
                data_labels=["True value", "Approximation"],
                colors=["blue", "orange"],
                x_label="date",
                y_label="value",
            )
        )

    def _visualize_testing(self, testing_set, prediction_set):
        # Testing data and prediction
        self.figures.append(
            visualize_data_series(
                title="Data Prediction",
                data_series=[testing_set, prediction_set],
                data_labels=["True prediction value", "Prediction"],
                colors=["blue", "red"],
                x_label="date",
                y_label="value",
            )
        )

    # Static method evaluating an Arima model
    @staticmethod
    def method_evaluation(
        order: Tuple[int, int, int],
        training_data: DataFrame,
        test_data: DataFrame,
        walk_forward: bool = True,
    ) -> List[float]:
        # Try catch block for numpy LU decomposition error
        try:
            # Create and fit model with training data
            arima_base = ARIMA(training_data, order=order)
            model = arima_base.fit()
            if not walk_forward:
                forecast = model.forecast(len(test_data))
                return forecast
            # Evaluate model with single step evaluation and Walk-forward validation
            forecast = []
            for index, row in test_data.iterrows():
                data_point = test_data.loc[index:index]
                prediction = Series(model.forecast(1))
                forecast.append(
                    prediction,  # Single step forecast
                )
                model = model.extend(data_point)
            forecast = DataFrame(pd.concat(DataFrame(x) for x in forecast))
            return forecast
        except:
            return None

    def __repr__(self):
        return f"<ArimaModel prouct_id: {self.get_name()}>"
