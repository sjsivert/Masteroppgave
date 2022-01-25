from pathlib import Path

import expects
from expects import be, be_false, be_true, expect, match, equal
from genpipes.compose import Pipeline
from mamba import after, before, description, it, _it
from matplotlib import pyplot as plt
from mockito import mock, unstub, when, ANY
from sklearn.linear_model import LogisticRegression
from spec.utils.test_data import test_data
from src.data_types.sklearn_model import SklearnModel
from src.data_types.validation_model import ValidationModel
from src.save_experiment_source.i_log_training_source import ILogTrainingSource
from src.save_experiment_source.local_checkpoint_save_source import LocalCheckpointSaveSource
from src.save_experiment_source.neptune_save_source import NeptuneSaveSource
from src.utils.file_hasher import generate_file_hash
from src.utils.temporary_files import temp_files

with description(NeptuneSaveSource, "api") as self:
    with before.all:
        # Using before.all instead of before.each here to cut down test time
        # and avoid too many Neptune API calls
        when(LocalCheckpointSaveSource, strict=False).write_file("neptune_id.txt", ANY(str))
        self.project_id = "sjsivertandsanderkk/test-project"
        options = {
            "project_id": self.project_id,
        }
        self.save_source = NeptuneSaveSource(
            **options,
            title="Test_experiment",
            description="Experiment for automated testing",
            sync=True,
        )

    with after.all:
        unstub()
        self.save_source.close()

    with it("can upload options, then fetch correct data"):
        options_data = "option1 option2\noption3 option4"
        self.save_source._save_options(options_data)
        fetched_options_data = self.save_source._load_options()
        expect(fetched_options_data).to(equal(options_data))

    with it("can upload and load models"):
        log_sources = [mock(ILogTrainingSource)]
        models = [
            ValidationModel(name="0", log_sources=log_sources),
            ValidationModel(name="1", log_sources=log_sources),
        ]
        self.save_source._save_models(models)
        # Load models of ValidationModel should set model variable model_loaded_content to given value
        self.save_source._load_models(models)
        for model in models:
            expect(model.model_loaded_contents).to(equal("Validation model. Mock model saving."))

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
            expect(self.save_source._verify_dataset_version(temp_datasets_path)).to(be_true)

    with it("can save metrics"):
        metrics_data = {"CPU": {"MAE": 5, "MSE": 6}, "GPU": {"MAE": 6, "MSE": 7}}
        self.save_source._save_metrics(metrics_data)

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

    with _it("can continue previous experiment"):
        first_run = NeptuneSaveSource(
            project_id=self.project_id,
            title="Test continue from last run",
            description="Test continue from last run",
        )
        metrics = {"CPU": {"MAE": 1.0, "MSE": 1.0}, "GPU": {"MAE": 1.0, "MSE": 1.0}}
        first_run._save_metrics(metrics)
        run_id = first_run.run.get_run_url().split("/")[-1]
        first_run.close()

        second_run = NeptuneSaveSource(
            project_id=self.project_id,
            title="Test continue from last run2",
            description="Test continue from last run2",
            load_from_checkpoint=True,
            neptune_id_to_load=run_id,
        )
        loaded_metrics = second_run.run["metrics"].fetch()
        expect(loaded_metrics["CPU"]["MAE"]).to(expects.equal(1.0))
