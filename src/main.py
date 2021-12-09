from __future__ import annotations

import logging

import click

from src.experiment import Experiment
from src.pipelines import market_insight_pipelines as pipelines
from src.utils import logger
from src.utils.config_parser import config


@click.command()
@click.option("--experiment", "-e", nargs=2, help="Experiment title and description.")
@click.option("--is-custom-run", is_flag=True)
def main(experiment, is_custom_run: bool):
    logger.init_logging()
    logging.info("Started")

    if experiment:
        logging.info(f'Starting experiment: "{experiment[0]}": "{experiment[1]}"')
        config["experiment_title"] = experiment[0]
        config["experiment_description"] = experiment[1]

        if is_custom_run:
            custom_run()

        experiment = Experiment(title=experiment[0], description=experiment[1])

        experiment.choose_model_structure(config["model"].get())

        experiment.load_and_process_data(pipelines.market_insight_pipeline())

        experiment.train_model()
        experiment.test_model()
        experiment.save_model()

    logging.info("Finished")


@click.command()
@click.option("--experiment", "-e", nargs=2, help="Experiment title and description.")
@click.option("--is-custom-run", is_flag=True)
@click.option(
    "--data-path",
    help="Path to preprocessed data. Implies skipping preprocessing of data",
)
@click.option(
    "--parameters",
    help="Path to hyperparameters to load the model with. Implies skipping hyperparameter grid search",
)
@click.option("--model-path", help="Path to a pretrained model. Implies skipping model training")
@click.option("--test-model", default=True)
@click.option("--save-results", default=True)
def custom_run(
    experiment,
    is_custom_run,
    data_path,
    parameters,
    model_path,
    test_model: bool,
    save_results: bool,
):
    """
    ./main.py --experiment 'title', 'description'
    ./main.py --custom-run='preprocess_data,parameter-search|train,test,save'
    ./main.py --custom-run='train,test,save' --data-path=<path>.csv --parameters=<path>.json
    ./main.py --custom-run='test|save' --data-path=<path>.csv --model-path=<path>
    ./main.py --continue-from-checkpoint
    """
    logging.info(
        f"Running as custom run with: \n\
    data_path: {data_path}\n\
    parameters: {parameters}\n\
    model_path: {model_path}\n\
    test_model: {test_model}\n\
    save_results: {save_results}\n\
    "
    )
    # TODO: Implement custom_run()


if __name__ == "__main__":
    main()
