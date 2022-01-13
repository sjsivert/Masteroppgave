from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import click

from src.experiment import Experiment
from src.pipelines import market_insight_preprocessing_pipeline as pipeline
from src.utils import logger
from src.utils.config_parser import config


@click.command()
@click.option(
    "--experiment", "-e", nargs=2, help="Experiment title and description. Title must be unique."
)
@click.option(
    "--save/--no-save",
    default=True,
    help="Boolean flag for saving the results or not. Overrides config.yaml.",
)
@click.option(
    "--continue-experiment", "-c", is_flag=True, help="Continues the last experiment executed."
)
def main(experiment: Tuple[str, str], save: bool, continue_experiment: bool) -> int:
    logger.init_logging()
    logging.info("Started")

    if experiment:
        logging.info(f'Starting experiment: "{experiment[0]}": "{experiment[1]}"')
        experiment = Experiment(
            title=experiment[0],
            description=experiment[1],
            save_sources_to_use=config["experiment"]["save_sources_to_use"].get() if save else [],
            save_source_options=config["experiment"]["save_source"].get() if save else {},
        )

        if save:
            experiment.run_complete_experiment(
                model_options=config["model"].get(),
                data_pipeline=pipeline.market_insight_pipeline(),
                save=True,
                options_to_save=config.dump(),
            )
        else:
            experiment.run_complete_experiment(
                model_options=config["model"].get(),
                data_pipeline=pipeline.market_insight_pipeline(),
                save=False,
            )
    elif continue_experiment:
        logging.info(f"Continues previous experiment")

        experiment_checkpoints_location = Path(
            config["experiment"]["save_source"]["disk"]["checkpoint_save_location"].get()
        )

        Experiment.continue_experiment(experiment_checkpoints_location)

    logging.info("Finished")
    return 0


if __name__ == "__main__":
    main()
