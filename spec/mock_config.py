from src.utils.config_parser import config


def init_mock_config():
    config.clear()
    config.read(user=False)
    config["experiment"] = {
        "save_sources_to_use": ["disk"],
        "save_source": {
            "disk": {
                "checkpoint_save_location": "./models/0_current_model_checkpoints/",
                "model_save_location": "./models/temp",
            },
            "neptune": {"project_id": "sjsivertandsanderkk/Masteroppgave"},
        },
    }
    config["experiment"]["save_sources_to_use"] = []
    config["experiment"]["save_source"]["disk"][
        "checkpoint_save_location"
    ] = "./models/0_current_model_checkpoints/"

    config["model"] = {
        "model_type": "validation_model",
        "rng_seed": 42,
        "local_univariate_arima": {
            "order": (1, 1, 1),
        },
    }
    config["data"] = {
        "data_path": "./datasets/raw/market_insights_overview_5p.csv",
        "categories_path": "./datasets/raw/solr_categories_2021_11_29.csv",
    }
