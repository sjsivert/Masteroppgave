import os

import pytest
from confuse.exceptions import NotFoundError
from expects import be_true, expect
from mamba import description, it
from src.utils.config_parser import config, get_absolute_path

with description("config_parser") as self:
    with it("get_absolute_path() finds config file"):
        expect(os.path.isfile(get_absolute_path("config.yaml"))).to(be_true)

    with it("exists"):
        expect(config.exists()).to(be_true)

    with it("fails on non-existent config parameter"):
        with pytest.raises(NotFoundError):
            config["non_existent_parameter"].get()
