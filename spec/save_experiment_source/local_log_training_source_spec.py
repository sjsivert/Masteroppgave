import os
from pathlib import Path

from expects import expect, be, be_true, match
from mamba import description, it, shared_context, included_context
from pandas import DataFrame

from src.save_experiment_source.local_log_training_source import LocalLogTrainingSource
from src.utils.temporary_files import temp_files

with description(LocalLogTrainingSource, "unit") as self:
    with shared_context("initialize_model"):
        save_location = "models/temp-log-training-source"
        with temp_files(save_location):
            experiment_title = "test-local-log-training-source"
            save_location = Path(save_location)
            log_location = f"{save_location}/{experiment_title}/logging"
            log_source = LocalLogTrainingSource(
                model_save_location=save_location,
                title=experiment_title,
            )
    with it("can be initialized"):
        with included_context("initialize_model"):
            pass

    with it("logs metrics to correct file"):
        with included_context("initialize_model"):
            # Arrange
            training_errors = {
                "GPU": {"MAE:": 42, "MSE": 0.69},
                "Nettverkskabler": {"MAE:": 420, "MSE": 1},
            }
            """
            epoch, cat_id_1, cat_id_2, cat_id_3,
            0
            """
            """
            ./models/test-model/logging/training_errors.csv
            """
            """
            epoch, model, MSE, MAE
            1, 202, 42, 0.69
            1, 203, 40, 0.60
            """
            log_source.log_metrics(training_errors)
            expect(os.path.isdir(f"{log_location}")).to(be_true)
            expect(os.path.isfile(f"{log_location}/training_errors.csv")).to(be_true)

            loaded_file = DataFrame.from_csv(f"{log_location}/training_errors.csv")
            expect(loaded_file.to_dict()).to(match(training_errors))

    with it("can log models"):
        pass
