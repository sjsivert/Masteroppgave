import pickle

from src.data_types.model import Model


class SklearnModel(Model):
    """
    Wrapper for sklearn models.
    """

    def __init__(self, model: object) -> None:
        self.model = model

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @staticmethod
    def load(path: str) -> Model:
        with open(path, "rb") as f:
            return SklearnModel(pickle.load(f))
