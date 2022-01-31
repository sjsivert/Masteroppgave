from src.utils.config_parser import config


def init_mock_config(
    model_struct_type: str = "validation_model", model_save_location: str = "./models/temp"
):
    config.clear()
    config.read(user=False)
    config.set(
        {
            "experiment": {
                "tags": ["tag3", "tag4"],
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
                    "training_size": 0.8,
                    "model_structure": [{"time_series_id": 11573, "order": (1, 1, 1)}],
                },
            },
            "data": {
                "data_path": "./README.md",
                "categories_path": "./README.md",
            },
        }
    )
