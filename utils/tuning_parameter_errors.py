from typing import Dict, List, Tuple

import numpy
import yaml

from sort_tuning_metrics import load_dict

# Global variables
experiment_path = "./models/"
experiment_name = "arima_corr_20_tuning_part1"


# TODO: Get tuning range from config
def get_parameter_range_config() -> Dict[str, Tuple[int, int]]:
    params = {}
    with open(f"{experiment_path}{experiment_name}/options.yaml") as f:
        config = yaml.full_load(f)
        params = config["model"]["local_univariate_arima"]["hyperparameter_tuning_range"]
    # params = {"p": (1,8), "d": (1,10), "q": (1,16)}
    return params


"""
:returns-1: Int value of number of parameters that should have been tuned
:returns-2: Int value of number of missing parameters that should have been tuned
"""


def get_number_parameter_tunings_arima() -> Dict[str, int]:
    parameter_error_dict = load_dict(
        f"{experiment_path}{experiment_name}/logging/tuning_metrics.csv"
    )
    parameter_error_dict = dict((x, len(list(y))) for x, y in parameter_error_dict.items())
    return parameter_error_dict


if __name__ == "__main__":
    parameter_range = get_parameter_range_config()
    number_of_parameters: int = numpy.prod([y[1] - y[0] + 1 for x, y in parameter_range.items()])
    params_nr = get_number_parameter_tunings_arima()
    missing = dict((x, number_of_parameters - y) for x, y in params_nr.items())

    # Printing the number of missing values

    print("Missing values")
    [print(f"{x}: {y}") for x, y in missing.items()]
    print()

    missing_pres = dict((x, y / number_of_parameters) for x, y in missing.items())
    print("Missing percentage")
    [print(f"{x}: {y}") for x, y in missing_pres.items()]
