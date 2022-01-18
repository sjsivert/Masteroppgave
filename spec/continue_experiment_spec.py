from contextlib import contextmanager
from pathlib import Path

import pytest
from expects import expect, match
from mamba import describe, it, description

from spec.mock_config import init_mock_config
from src.continue_experiment import ContinueExperiment
from src.utils.config_parser import config
from src.utils.temporary_files import temp_files


@contextmanager
def temp_checkpoints_location(temp_dir: str):
    with temp_files(temp_dir):
        # Arrange
        with open(f"{temp_dir}/title-description.txt", "w") as f:
            f.write("title\ndescription")

        with open(f"{temp_dir}/options.yaml", "w") as f:
            init_mock_config()
            f.write(config.dump())

        yield


with description(ContinueExperiment, "unit") as self:
    with it("initializes correctly with title and description"):
        # Arrange
        temp_location = "temp_continue_experiment"
        with temp_checkpoints_location(temp_location):
            # Act
            experiment = ContinueExperiment(Path(temp_location))
            experiment.continue_experiment()

            # Assert
            expect(experiment.title).to(match("title"))
            expect(experiment.description).to(match("description"))

    with it("fails if no checkpoint file is found"):
        with pytest.raises(AssertionError):
            ContinueExperiment(Path("tmp/not_found"))

    with it("fails if no title-description.txt file isf found"):
        temp_location = "temp_continue_experiment_no_title_description"
        with temp_files(temp_location):
            experiment = ContinueExperiment(Path(temp_location))
            with pytest.raises(FileNotFoundError):
                experiment.continue_experiment()
