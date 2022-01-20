from genpipes.compose import Pipeline
from mamba import after, before, description, it
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

from spec.utils.test_data import test_data
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
        self.save_source._save_options("option1 option2")

    with it("can upload models"):
        models = [SklearnModel(LogisticRegression()), SklearnModel(LogisticRegression())]
        self.save_source._save_models(models)

    with it("can save metrics"):
        self.save_source._save_metrics({"CPU": {"MAE": 5, "MSE": 6}, "GPU": {"MAE": 6, "MSE": 7}})

    with it("can can save figures"):
        # Arrange
        fig, ax = plt.subplots()
        data = [1, 2, 3, 4, 5]
        ax.plot(data)
        ax.set_title("Test_title")
        self.save_source._save_figures([fig])

    with it("can run save_model_and_metadata() without crashing"):
        self.save_source.save_model_and_metadata(
            options="options",
            metrics={},
            models=[],
            figures=[],
            data_pipeline_steps=Pipeline(steps=[("load test data", test_data)]).__str__(),
        )
