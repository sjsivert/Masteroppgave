from expects import expect, equal, be_true
from mamba import describe, it, description

from spec.mock_config import init_mock_config
from src.utils.config_parser import config
from src.utils.extract_tags_from_config import extract_tags_from_config

with description(extract_tags_from_config, "unit") as self:
    with it("it returns [ValidationModel] from mock_config"):
        init_mock_config()
        tags = extract_tags_from_config()

        expect(tags).to(equal(["validation_model"]))
