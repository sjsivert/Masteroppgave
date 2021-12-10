from confuse.exceptions import NotFoundError
from expects import be_true, expect
from mamba import description, it
from mockito import mock, when
from src.utils.config_parser import get_absolute_path
from src.utils.logger import init_logging

with description("main.py") as self:
    with it("runs without parameters"):
        expect(True).to(be_true)

    with it("executes init_logging"):
        mock_logger = mock(init_logging())
