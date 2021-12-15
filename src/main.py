from __future__ import annotations

import logging
from typing import Tuple

import click

from src.experiment import Experiment
from src.pipelines import market_insight_preprocessing_pipeline as pipeline
from src.utils import logger
from src.utils.config_parser import config


@click.command()
@click.option("--experiment", "-e", nargs=2, help="Experiment title and description. Title must be unique.")
@click.option("--is-custom-run", is_flag=True)
@click.option(
    "--save/--no-save", default=True, help="Boolean flag for saving the results or not. Overrides config.yaml."
)
def main(experiment: Tuple[str, str], is_custom_run: bool, save: bool) -> int:
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

        experiment.choose_model_structure(config["model"].get())

        experiment.load_and_process_data(pipeline.market_insight_pipeline())

        experiment.train_model()
        experiment.test_model()
        experiment.save_model(options=config.dump()) if save else None

    logging.info("Finished")
    return 0


if __name__ == "__main__":
    main()
