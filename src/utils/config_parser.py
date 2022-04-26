#!/usr/bin/env python
from collections import OrderedDict
from pathlib import Path

import confuse

current_dir = Path(__file__)
root_dir = Path(current_dir.parent.parent.parent)

config = confuse.Configuration("Masteroppgave", __name__)
config.set_file(f"{root_dir}/config.yaml")
config.set_file(f"config/{config['model']['config']}.yaml")
config.set_env()
config.exists()


def get_absolute_path(path):
    return f"{root_dir}/{path}"


def update_config_lstm_params(config_value: dict):
    global_or_local = (
        "local_model"
        if config["model"]["model_type"].get() == "local_univariate_lstm"
        else "global_model"
    )
    old_config = config["model"]["model_config"][global_or_local]["model_structure"].get()
    print("old_config", old_config)
    converted_config_value = convert_numbered_config_items_to_list(config_value)
    print("new_cofnig", converted_config_value)
    config["model"]["model_config"][global_or_local]["model_structure"].set(
        [converted_config_value]
    )


def convert_numbered_config_items_to_list(items: dict):
    layers = []
    for layer in range(items["number_of_layers"]):
        layer_params = OrderedDict(
            {
                "hidden_size": items.pop(f"hidden_size_{layer}"),
                "dropout": items.pop(f"dropout_{layer}"),
                "recurrent_dropout": items.pop(f"recurrent_dropout_{layer}"),
            }
        )
        layers.append(layer_params)
    items.update({"layers": layers})
    return items
