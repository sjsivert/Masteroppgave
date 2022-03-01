from mamba import description, before, after, shared_context, it, included_context
from mockito import unstub, mock

from spec.mock_config import init_mock_config
from src.model_strutures.local_univariate_lstm_structure import LocalUnivariateLstmStructure
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from src.utils.config_parser import config

with description(LocalUnivariateLstmStructure, "unit") as self:
    with before.all:
        init_mock_config()
        self.options = config["model"].get()

    with after.each:
        unstub()

    with shared_context("mocks"):
        log_source = mock(ILogTrainingSource)
    with it("Can be initialised with options"):
        with included_context("mocks"):
            model = LocalUnivariateLstmStructure(
                log_sources=[log_source], **self.options["local_univariate_lstm"]
            )
