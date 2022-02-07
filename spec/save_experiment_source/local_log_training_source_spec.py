import os
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from expects import expect, be, be_true, match, equal
from mamba import description, it, shared_context, included_context, _it, before
from pandas import DataFrame

from src.save_experiment_source.local_log_training_source import LocalLogTrainingSource
from src.utils.temporary_files import temp_files


@contextmanager
def temp_log_training_files(path: Path):
    with temp_files(path.__str__()):
        yield


with description(LocalLogTrainingSource, "unit") as self:
    with before.each:
        self.save_location = Path("models/temp-log-training-source")
        self.experiment_title = "test-local-log-training-source"
        self.log_location = f"{self.save_location}/{self.experiment_title}/logging"

    with it("can be initialized"):
        with temp_log_training_files(self.save_location):
            log_source = LocalLogTrainingSource(
                model_save_location=self.save_location,
                title=self.experiment_title,
            )

    with it("logs metrics to correct file"):
        with temp_log_training_files(self.save_location):
            log_source = LocalLogTrainingSource(
                model_save_location=self.save_location,
                title=self.experiment_title,
            )
            # Arrange
            training_errors = {
                "GPU": {"MAE": 42, "MSE": 0.69},
                "Nettverkskabler": {"MAE": 420, "MSE": 1},
            }
            # ./models/test-model/logging/training_errors.csv

            expected_file_content = DataFrame(
                {
                    "epoch": [0, 0, 1],
                    "model_id": ["GPU", "Nettverkskabler", "GPU"],
                    "MAE": [42, 420, 3],
                    "MSE": [0.69, 1, 4],
                }
            )

            log_source.log_metrics(training_errors)
            expect(os.path.isdir(f"{self.log_location}")).to(be_true)
            expect(os.path.isfile(f"{self.log_location}/training_errors.csv")).to(be_true)
            print(pd.read_csv(f"{self.log_location}/training_errors.csv"))

            # Log again to check it append the file
            log_source.log_metrics({"GPU": {"MAE": 3, "MSE": 4}})

            loaded_file = pd.read_csv(f"{self.log_location}/training_errors.csv")

            expect(loaded_file.__str__()).to(match(expected_file_content.__str__()))

    with it("can log models"):
        # TODO: Implement
        pass

    with it("will extract correct error metrics form a metrics dictionary"):
        training_errors = {
            "GPU": {"MAE": 43, "MSE": 0.69},
            "Nettverkskabler": {"MAE": 420, "MSE": 1, "test": 37},
        }
        error_metrics = LocalLogTrainingSource.extract_all_error_metrics_from_dict(training_errors)
        expect(error_metrics).to(equal(["MAE", "MSE", "test"]))
