import os
import shutil

from expects import be_true, equal, expect, match
from mamba import after, before, description, it
from spec.mock_config import init_mock_config
from src.save_experiment_source.local_checkpoint_save_source import LocalCheckpointSaveSource

with description(LocalCheckpointSaveSource, "unit") as self:
    with before.each:
        init_mock_config()
        self.checkpoint_save_location = LocalCheckpointSaveSource().get_checkpoint_save_location()

    with after.each:
        shutil.rmtree(LocalCheckpointSaveSource().get_checkpoint_save_location())

    with it("wipes and create checkpoint location as expected"):
        LocalCheckpointSaveSource.wipe_and_init_checkpoint_save_location()
        expect(self.checkpoint_save_location.is_dir()).to(be_true)
        expect(len(os.listdir(self.checkpoint_save_location))).to(equal(0))

        LocalCheckpointSaveSource.write_file("options.yaml", "options yo")
        LocalCheckpointSaveSource.write_file("title-description.txt", "title yo, description yo")

        # Asses
        expect(self.checkpoint_save_location.__str__()).to(match("models/temp-checkpoints"))
        expect(self.checkpoint_save_location.is_dir()).to(be_true)
        expect(self.checkpoint_save_location.joinpath("options.yaml").is_file()).to(be_true)
        expect(self.checkpoint_save_location.joinpath("title-description.txt").is_file()).to(
            be_true
        )

    with it("it loads title and description as expected"):
        LocalCheckpointSaveSource.wipe_and_init_checkpoint_save_location()
        title = "title yo"
        description = "description yo"
        LocalCheckpointSaveSource.write_file("title-description.txt", f"{title}\n{description}")
        title, description = LocalCheckpointSaveSource.load_title_and_description()
        expect(title).to(equal(title))
        expect(description).to(equal(description))
