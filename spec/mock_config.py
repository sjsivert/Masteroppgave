from src.utils.config_parser import config


def init_mock_config(
    model_struct_type: str = "validation_model", model_save_location: str = "./models/temp"
):
    config.clear()
    config.read(user=False)
    config.set(
        {
            "logger": {
                "log_level": "ERROR",
                "log_file": "",
            },
            "experiment": {
                "tags": ["tag3", "tag4"],
                "error_metrics": ["MSE", "MAE", "MASE", "SMAPE"],
                "save_sources_to_use": ["disk"],
                "checkpoint_save_location": "./models/temp-checkpoints",
                "log_model_every_n_epoch": 1,
                "save_source": {
                    "disk": {
                        "model_save_location": model_save_location,
                    },
                    "neptune": {"project_id": "sjsivertandsanderkk/Masteroppgave"},
                },
            },
            "model": {
                "model_type": model_struct_type,
                "rng_seed": 42,
                "local_univariate_arima": {
                    "metric_to_use_when_tuning": "MASE",
                    "forecast_window_size": 7,
                    "steps_to_predict": 2,
                    "multi_step_forecast": False,
                    "model_structure": [
                        {"time_series_id": 11573, "hyperparameters": {"p": 1, "d": 1, "q": 1}}
                    ],
                },
                "local_univariate_lstm": {
                    "common_parameters_for_all_models": {
                        "training_size": 0.8,
                        "input_window_size": 7,
                        "output_window_size": 1,
                        "batch_size": 32,
                        "optimizer_name": "SGD",
                        "number_of_epochs": 1,
                    },
                    "hyperparameter_tuning_range": {
                        "number_of_tuning_trials": 10,
                        "hidden_size": [10, 200],
                        "number_of_layers": [1, 5],
                        "dropout": [0.0, 0.5],
                        "optimizer_name": ["Adam", "SGD", "RMSprop"],
                        "learning_rate": [1e-5, 1e-1],
                        "number_of_epochs": [1, 2],
                        "input_window_size": 1,
                        "output_window_size": 1,
                        "number_of_features": 1,
                        "number_of_trials": 1,
                        "batch_size": [1, 32],
                    },
                    "model_structure": [
                        {
                            "time_series_id": 12532,
                            "learning_rate": 0.001,
                            "hidden_layer_size": 100,
                            "dropout": 0.1,
                            "number_of_features": 1,
                            "number_of_layers": 1,
                        },
                        {
                            "time_series_id": 11694,
                            "learning_rate": 0.001,
                            "hidden_layer_size": 100,
                            "dropout": 0.1,
                            "number_of_features": 1,
                            "number_of_layers": 2,
                        },
                    ],
                },
            },
            "data": {
                "data_path": "./README.md",
                "categories_path": "./README.md",
            },
        }
    )
