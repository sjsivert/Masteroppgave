from mamba import after, before, description, it
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from src.data_types.sklearn_model import SklearnModel
from src.save_experiment_source.neptune_save_source import NeptuneSaveSource

with description(NeptuneSaveSource, "api") as self:
    with before.all:
        options = {
            "project_id": "sjsivertandsanderkk/test-project",
        }
        self.save_source = NeptuneSaveSource(
            **options,
            title="Test_experiment",
            description="Experiment for automated testing",
            tags=["test"],
        )

    with after.all:
        self.save_source.close()

    with it("can upload options"):
        self.save_source.save_options("option1 option2")

    with it("can upload models"):
        models = [SklearnModel(LogisticRegression()), SklearnModel(LogisticRegression())]
        self.save_source.save_models(models)

    with it("can save metrics"):
        self.save_source.save_metrics(["mae: 0.1", "mse: 0.2"])

    with it("can can save figures"):
        # Arrange
        fig, ax = plt.subplots()
        data = [1, 2, 3, 4, 5]
        ax.plot(data)
        ax.set_title("Test_title")
        self.save_source.save_figures([fig])
