from src.utils.config_parser import config


def init_mock_config(
    model_struct_type: str = "validation_model", model_save_location: str = "./models/temp"
):
    config.clear()
    config.read(user=False)
    config["experiment"] = {
        "save_sources_to_use": ["disk"],
        "save_source": {
            "disk": {
                "checkpoint_save_location": "./models/0_current_model_checkpoints/",
                "model_save_location": model_save_location,
            },
            "neptune": {"project_id": "sjsivertandsanderkk/Masteroppgave"},
        },
    }
    config["experiment"]["save_sources_to_use"] = ["disk"]
    config["experiment"]["save_source"]["disk"][
        "checkpoint_save_location"
    ] = "./spec/0_current_model_checkpoints/"
    config["experiment"]["save_source"]["disk"]["log_model_every_n_epoch"] = 1
    config["experiment"]["save_source"]["disk"]["model_save_location"] = model_save_location

    config["model"] = {
        "model_type": model_struct_type,
        "rng_seed": 42,
        "local_univariate_arima": {
            "order": (1, 1, 1),
        },
    }
    config["data"] = {
        "data_path": "./README.md",
        "categories_path": "./README.md",
    }
