import os

from metrics_value_extraction import *

#################################################################
#################################################################
######### Create tables with student t-test info ################
#################################################################
#################################################################


# Main experiments (datasets 1, 2 and 3)

print("Main")
dataset = "dataset_1"
projects_1 = {
    "local univariate lstm": f"{dataset}-lstm-local-univariate-tune-400-trials",
    "local multivariate lstm": f"{dataset}-lstm-local-multivariate-tune-400-trials",
    "global univariate lstm": f"{dataset}-lstm-global-univariate-tune-400-trials",
    "global multivariate lstm": f"{dataset}-lstm-global-multivariate-tune-400-trials",

    "local univariate cnn ae lstm": f"{dataset}-cnn-ae-lstm-local-univariate",
    "local multivariate cnn ae lstm": f"{dataset}-cnn-ae-lstm-local-multivariate",
    "global univariate cnn ae lstm": f"{dataset}-cnn-ae-lstm-global-univariate",
    "global multivariate cnn ae lstm": f"{dataset}-cnn-ae-lstm-global-multivariate",
}

dataset = "dataset_2"
projects_2 = {
    "local univariate lstm": f"{dataset}-lstm-local-univariate-tune-400-trials",
    "local multivariate lstm": f"{dataset}-lstm-local-multivariate-tune-400-trials",
    "global univariate lstm": f"{dataset}-lstm-global-univariate-tune-400-trials-exp",
    "global multivariate lstm": f"{dataset}-lstm-global-multivariate-tune-400-trials",

    "local univariate cnn ae lstm": f"{dataset}-cnn-ae-lstm-local-univariate",
    "local multivariate cnn ae lstm": f"{dataset}-cnn-ae-lstm-local-multivariate",
    "global univariate cnn ae lstm": f"{dataset}-cnn-ae-lstm-global-univariate",
    "global multivariate cnn ae lstm": f"{dataset}-cnn-ae-lstm-global-multivariate",
}

dataset = "dataset_seasonal"
projects_3 = {
    "local univariate lstm": "dataset_seasonal-lstm-local-univariate-final",
    "local multivariate lstm": "dataset_seasonal-lstm-local-multivariate-final",
    "global univariate lstm": "dataset_seasonal-lstm-global-univariate-final",
    "global multivariate lstm": "dataset_seasonal-lstm-global-multivariate-final",

    "local univariate cnn ae lstm": f"{dataset}-cnn-ae-lstm-local-univariate",
    "local multivariate cnn ae lstm": f"{dataset}-cnn-ae-lstm-local-multivariate",
    "global univariate cnn ae lstm": f"{dataset}-cnn-ae-lstm-global-univariate",
    "global multivariate cnn ae lstm": f"{dataset}-cnn-ae-lstm-global-multivariate",
}


table_save_path = f"./MastersThesis/tables/results/ttest"
if not os.path.isdir(table_save_path):
    os.mkdir(table_save_path)

test_significanse_multiple_datasets(
    [projects_1, projects_2, projects_3],
    ["dataset 1", "dataset 2", "dataset 3"],
    "sMAPE",
    name="main-experiments-sMAPE",
    tabel_text="Student t-test, measuring confidence of significant difference between predictions on the CNN-AE-LSTM and the LSTM for different model structures. sMape error",
    table_save_path=table_save_path
)
test_significanse_multiple_datasets(
    [projects_1, projects_2, projects_3],
    ["dataset 1", "dataset 2", "dataset 3"],
    "MASE",
    name="main-experiments-MASE",
    tabel_text="Student t-test, measuring confidence of significant difference between predictions on the CNN-AE-LSTM and the LSTM for different model structures. MASE error",
    table_save_path=table_save_path
)







# LSTM experiments  TODO:
print("LSTM")
dataset = "dataset_1"
projects_1 = {
    "local univariate lstm": f"{dataset}-lstm-local-univariate-tune-400-trials",
    "local multivariate lstm": f"{dataset}-lstm-local-multivariate-tune-400-trials",
    "global univariate lstm": f"{dataset}-lstm-global-univariate-tune-400-trials",
    "global multivariate lstm": f"{dataset}-lstm-global-multivariate-tune-400-trials",
}
test_significanse_each_experiment(
    projects_1,
    "sMAPE",
    name="lstm-experiments-sMAPE-dataset-1",
    tabel_text="Student t-test, measuring confidence of significant difference between the local Univariate LSTM and other LSTM models on dataset 1. sMape error",
    table_save_path=table_save_path
)
test_significanse_each_experiment(
    projects_1,
    "MASE",
    name="lstm-experiments-MASE-dataset-1",
    tabel_text="Student t-test, measuring confidence of significant difference between the local Univariate LSTM and other LSTM models on dataset 1. MASE error",
    table_save_path=table_save_path
)

dataset = "dataset_2"
projects_2 = {
    "local univariate lstm": f"{dataset}-lstm-local-univariate-tune-400-trials",
    "local multivariate lstm": f"{dataset}-lstm-local-multivariate-tune-400-trials",
    "global univariate lstm": f"{dataset}-lstm-global-univariate-tune-400-trials-exp",
    "global multivariate lstm": f"{dataset}-lstm-global-multivariate-tune-400-trials",
}
test_significanse_each_experiment(
    projects_2,
    "sMAPE",
    name="lstm-experiments-sMAPE-dataset-2",
    tabel_text="Student t-test, measuring confidence of significant difference between LSTM models dataset 2, statistic value. sMape error",
    table_save_path=table_save_path
)
test_significanse_each_experiment(
    projects_2,
    "MASE",
    name="lstm-experiments-MASE-dataset-2",
    tabel_text="Student t-test, measuring confidence of significant difference between LSTM models dataset 2, statistic value. MASE error",
    table_save_path=table_save_path
)

projects_3 = {
    "local univariate lstm": "dataset_seasonal-lstm-local-univariate-final",
    "local multivariate lstm": "dataset_seasonal-lstm-local-multivariate-final",
    "global univariate lstm": "dataset_seasonal-lstm-global-univariate-final",
    "global multivariate lstm": "dataset_seasonal-lstm-global-multivariate-final",
}
test_significanse_each_experiment(
    projects_3,
    "sMAPE",
    name="lstm-experiments-sMAPE-dataset-3",
    tabel_text="Student t-test, measuring confidence of significant difference between LSTM models dataset 3, statistic value. sMape error",
    table_save_path=table_save_path
)
test_significanse_each_experiment(
    projects_3,
    "MASE",
    name="lstm-experiments-MASE-dataset-3",
    tabel_text="Student t-test, measuring confidence of significant difference between LSTM models dataset 3, statistic value. MASE error",
    table_save_path=table_save_path
)









# Differnencing data
print("Diff")
dataset = "Differencing dataset"
projects = {
    "local univariate lstm dataset 1": "dataset_1-lstm-local-univariate-tune-400-trials",
    "local univariate lstm seasonal": "dataset_seasonal-lstm-local-univariate-tune-400-trials",
    "local univariate lstm dataset 1 diff": "dataset_1-lstm-local-unviariate-tune-400-trials-with-differencing",
    "local univariate lstm seasonal diff": "dataset_seasonal-lstm-local-univariate-differencing-tune-400-trials",
}
test_significanse_multiple_datasets(
    [projects],
    [dataset],
    "MASE",
    name="differencing-experiments-MASE",
    tabel_text="Student t-test, measuring confidence of significant difference between predictions, statistic value. MASE error",
    table_save_path=table_save_path
)



print("Variancce")
# Experiment variance / noise data
high_variance_experiments = {
    "high-variance-lstm-local-univariate": "dataset-high-variance-lstm-local-univariate",
    "high-variance-cnn-ae-lstm-local-univariate": "dataset-high-variance-cnn-ae-lstm-local-univariate",
}
ok_variance_experiments = {
    "ok-variance-lstm-local-univariate": "dataset-ok-variance-lstm-local-univariate",
    "ok-variance-local-univariate-cnn-ae-lstm": "dataset-ok-variance-cnn-ae-lstm-local-univariate",
}

low_variance_experiments = {
    "low-variance-lstm-local-univariate": "dataset-low-variance-lstm-local-univariate",
    "low-variance-cnn-ae-lstm-local-univariate": "dataset-low-variance-cnn-ae-lstm-local-univariate",
}

test_significanse_multiple_datasets(
    [high_variance_experiments, ok_variance_experiments, low_variance_experiments],
    ["High variance", "OK variance", "Low variance"],
    "sMAPE",
    name="variance-experiments-sMAPE",
    tabel_text="Student t-test, measuring confidence of significant difference between predictions, statistic value. sMape error",
    table_save_path=table_save_path
)
test_significanse_multiple_datasets(
    [high_variance_experiments, ok_variance_experiments, low_variance_experiments],
    ["High variance", "OK variance", "Low variance"],
    "MASE",
    name="variance-experiments-MASE",
    tabel_text="Student t-test, measuring confidence of significant difference between predictions, statistic value. MASE error",
    table_save_path=table_save_path
)

