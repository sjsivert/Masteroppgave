# fmt: off
import optuna
from mamba import description, it

from src.utils.temporary_files import temp_files

with description("OptunaStorageExploration", "unit") as self:
    with it("should create a study with sql lite storage"):
        temp_dir = "test-optuna-storage"
        with temp_files(temp_dir):

            def objective(trial):
                x = trial.suggest_float("x", 0, 10)
                return x**2

            sqlite_connection_string = f"sqlite:///{temp_dir}/test.db"
            study = optuna.create_study(storage=sqlite_connection_string, study_name="test")
            study.optimize(objective, n_trials=3)

            loaded_study = optuna.load_study(study_name="test", storage=sqlite_connection_string)
            assert len(loaded_study.trials) == len(study.trials)
