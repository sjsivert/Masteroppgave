from typing import List, Dict, Tuple
from matplotlib.figure import Figure
from pandas import DataFrame
from src.data_types.i_model import IModel
from src.save_experiment_source.i_log_training_source import ILogTrainingSource

from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.utils.error_calculations import calculate_error
from src.utils.visuals import visualize_data_series


class ArimaModel(IModel):

    # TODO! Set random SEED
    def __init__(
        self,
        log_sources: List[ILogTrainingSource],
        name: str = "placeholder",
        order: Tuple[int] = (0, 0, 0),
    ):
        super().__init__(log_sources=log_sources, name=name)
        # The ARIMA model must be instanced with data, thus this is done during training
        self.model = None
        self.order = order  # Tuple defining the ARIMA order variable
        self.training_periode = (
            0  # Integer defining the number of data-points used to train the ARIMA model
        )
        self.training_residuals = None  # Dataframe of training residuals

    def train(self, data_set: DataFrame) -> Dict:
        self.training_periode = len(data_set)
        arima_model = ARIMA(data_set, order=self.order)
        arima_model_res = arima_model.fit()
        self.model = arima_model_res
        self.value_approximation = self.model.predict(0, self.training_periode - 1)
        # Figures
        self._visualize_training(data_set, self.value_approximation)
        # Metrics
        metrics = calculate_error(data_set, self.value_approximation)
        self.metrics["Training_MSE"] = metrics["MSE"]
        self.metrics["Training_MAE"] = metrics["MAE"]
        return metrics

    def test(self, data_set: DataFrame, predictive_period: int = 5) -> Dict:
        predictive_period = (
            predictive_period if predictive_period <= len(data_set) else len(data_set)
        )
        value_predictions = self.model.predict(
            self.training_periode, self.training_periode + predictive_period - 1
        )
        self.predictions = DataFrame(value_predictions)
        # Figures
        self._visualize_testing(data_set[:predictive_period], self.predictions)
        # Metrics
        metrics = calculate_error(data_set[:predictive_period], self.predictions)
        self.metrics["Test_MSE"] = metrics["MSE"]
        self.metrics["Test_MAE"] = metrics["MAE"]
        return metrics

    def get_metrics(self) -> Dict:
        return self.metrics

    def get_figures(self) -> List[Figure]:
        return self.figures

    def save(self, path: str) -> str:
        save_path = f"{path}Arima_{self.get_name()}.pkl"
        self.model.save(save_path)
        return save_path

    def load(self, path: str) -> IModel:
        # TODO: Potential bug in loading method. Verify this in testing
        try:
            load_path = f"{path}Arima_{self.get_name()}.pkl"
            loaded_model = ARIMAResults.load(load_path)
            self.model = loaded_model
            return loaded_model
        except Exception as e:
            print("Load execption thrown:", e)  # TODO: Exception used for testing
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
                data_labels=["True value", "Prediction"],
                colors=["blue", "red"],
                x_label="date",
                y_label="value",
            )
        )
