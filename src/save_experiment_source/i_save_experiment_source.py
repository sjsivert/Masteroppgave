from typing import List


class ISaveExperimentSource:
    """
    An interface for all experiment save sources to implement.
    For example, disk, database, neptune.ai, etc.
    """

    def __init__(self) -> None:
        # Interface, not to be implemented
        pass

    def save_options(self, options: str) -> None:
        # Interface, not to be implemented
        pass

    def save_models(self, models: List) -> None:
        # Interface, not to be implemented
        pass

    def save_metrics(self, metrics: List) -> None:
        # Interface, not to be implemented
        pass
