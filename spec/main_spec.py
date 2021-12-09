from confuse.exceptions import NotFoundError
from expects import be_false, be_true, expect
from mamba import description, it
from mockito import mock, when
from src import main
from src.utils.config_parser import config, get_absolute_path
from src.utils.logger import init_logging

with description("main.py") as self:
    with it("runs without parameters"):
        # main.main()
        pass
    with it("executes init_logging"):
        mock_logger = mock(init_logging())
        # main.main()
        # expect(mock_logger.called).to(be_false)
