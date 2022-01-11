<<<<<<< Updated upstream
from mamba import after, before, description, it
from matplotlib import pyplot as plt
=======
import matplotlib.pyplot as plt
import numpy as np
from mamba import before, description, it
from matplotlib.figure import Figure
>>>>>>> Stashed changes
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

    with it("can upload figures"):
        figures = []
        # Create figures for upload
        t = np.arange(0.0, 2.0, 0.01)
        f1 = np.sin(2 * np.pi * t)
        f2 = np.cos(2 * np.pi * t)
        plt.figure(1)
        plt.plot(t, f1)
        plt.figure(2)
        plt.plot(t, f2)
        for i in plt.get_fignums():
            figures.append(plt.figure(i))
        # Upload figures
        self.save_source.save_figures(figures)

    with it("can upload models"):
        models = [SklearnModel(LogisticRegression()), SklearnModel(LogisticRegression())]
        self.save_source.save_models(models)

    with it("can save metrics"):
        self.save_source.save_metrics({"CPU": {"MAE": 5, "MSE": 6}, "GPU": {"MAE": 6, "MSE": 7}})

    with it("can can save figures"):
        # Arrange
        fig, ax = plt.subplots()
        data = [1, 2, 3, 4, 5]
        ax.plot(data)
        ax.set_title("Test_title")
        self.save_source.save_figures([fig])
