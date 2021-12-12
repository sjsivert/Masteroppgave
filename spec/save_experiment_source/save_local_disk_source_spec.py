import os
import shutil

import pytest
from expects import be, be_false, be_true, expect
from mamba import after, before, description, it
from src.save_experiment_source.save_local_disk_source import \
    SaveLocalDiskSource

with description("SaveLocalDiskSource") as self:
    with before.all:
        try:
            os.mkdir("spec/temp/")
        except FileExistsError:
            pass

    with after.all:
        shutil.rmtree("spec/temp/")
    
    with before.each:
        self.options = {"model_save_location": "spec/temp/"}
        self.save_source = SaveLocalDiskSource(self.options, "test_experiment")

    with after.each:
        shutil.rmtree("spec/temp/test_experiment")

    with it("Throws exception when save location does already exist"):
        os.mkdir("spec/temp/this-folder-exists")
        with pytest.raises(FileExistsError):
            save_source = SaveLocalDiskSource(self.options, "this-folder-exists")

    with it("Initializes correctly when save location does not exist"):
        options = {"model_save_location": "spec/temp/"}
        save_source = SaveLocalDiskSource(options, "this-folder-does-not-exist")
        assert save_source.save_location == "spec/temp/this-folder-does-not-exist"

    with it("save options as options.yaml inside correct folder"):
        self.save_source.save_options("option 1\noption2")
        expect(os.path.isfile("spec/temp/test_experiment/options.yaml")).to(be_true)

    with it("saves metrix as metrics.txt inside correct folder"):
        self.save_source.save_metrics(["mae: 0.1", "mse: 0.2"])
        expect(os.path.isfile("spec/temp/test_experiment/metrics.txt")).to(be_true)
