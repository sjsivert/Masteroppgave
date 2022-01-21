import expects
from expects import be, be_false, be_true, expect, match
from genpipes.compose import Pipeline
from mamba import after, before, description, it
from matplotlib import pyplot as plt
from mockito import mock
from sklearn.linear_model import LogisticRegression
from spec.utils.test_data import test_data
from src.data_types.sklearn_model import SklearnModel
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from src.save_experiment_source.neptune_save_source import NeptuneSaveSource
from src.utils.temporary_files import temp_files

with description(NeptuneSaveSource, "api") as self:
    with before.all:
        options = {
            "project_id": "sjsivertandsanderkk/test-project",
        }
        self.save_source = NeptuneSaveSource(
            **options,
            title="Test_experiment",
            description="Experiment for automated testing",
        )

    with after.all:
        self.save_source.close()

    with it("can upload options"):
        self.save_source._save_options("option1 option2")

    with it("can upload models"):
        log_sources = [mock(ILogTrainingSource)]
        models = [
            SklearnModel(model=LogisticRegression(), log_sources=log_sources),
            SklearnModel(LogisticRegression(), log_sources),
        ]
        self.save_source._save_models(models)

    with it("can track datasets"):
        model_data_save_location = "./models/temp_data"
        with temp_files(model_data_save_location):
            temp_datasets_path = {
                "raw_data": f"{model_data_save_location}/raw_data.csv",
                "category_data": f"{model_data_save_location}/category_data.csv",
            }
            for data_path_name, data_path in temp_datasets_path.items():
                with open(data_path, "w") as f:
                    f.write(f"Test data containing data from {data_path_name}")
            self.save_source._save_dataset_version(temp_datasets_path)

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
            datasets={},
            models=[],
            figures=[],
            data_pipeline_steps=Pipeline(steps=[("load test data", test_data)]).__str__(),
            experiment_tags=["test"],
        )

    with it("can continue previous experiment"):
        first_run = NeptuneSaveSource(
            project_id="sjsivertandsanderkk/test-project",
            title="Test continue from last run",
            description="Test continue from last run",
        )
        metrics = {"CPU": {"MAE": 1.0, "MSE": 1.0}, "GPU": {"MAE": 1.0, "MSE": 1.0}}
        first_run._save_metrics(metrics)
        run_id = first_run.run.get_run_url().split("/")[-1]
        first_run.close()

        second_run = NeptuneSaveSource(
            project_id="sjsivertandsanderkk/test-project",
            title="Test continue from last run2",
            description="Test continue from last run2",
            load_from_checkpoint=True,
            load_run_id=run_id,
        )
        loaded_metrics = second_run.run["metrics"].fetch()
        expect(loaded_metrics["CPU"]["MAE"]).to(expects.equal(1.0))
