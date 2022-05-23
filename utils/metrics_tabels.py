import os

from metrics_value_extraction import *

##############################################################
##############################################################
######  Extract data from metrics and write to tables  #######
##############################################################
##############################################################
figure_base_path = f"./MastersThesis/figs/results"


# Dataset 1
dataset = "dataset_1"
table_save_path = f"./MastersThesis/tables/results/{dataset}"
if not os.path.isdir(table_save_path):
    os.mkdir(table_save_path)

projects = {
    "sarima": f"{dataset}-sarima",
    "local univariate lstm": f"{dataset}-lstm-local-univariate-tune-400-trials",
    "local multivariate lstm": f"{dataset}-lstm-local-multivariate-tune-400-trials",
    "global univariate lstm": f"{dataset}-lstm-global-univariate-tune-400-trials",
    "global multivariate lstm": f"{dataset}-lstm-global-multivariate-tune-400-trials",

    "local univariate cnn ae lstm": f"{dataset}-cnn-ae-lstm-local-univariate",
    "local multivariate cnn ae lstm": f"{dataset}-cnn-ae-lstm-local-multivariate",
    "global univariate cnn ae lstm": f"{dataset}-cnn-ae-lstm-global-univariate",
    "global multivariate cnn ae lstm": f"{dataset}-cnn-ae-lstm-global-multivariate",
}
create_all_tables_each_experiment(projects, save_path=table_save_path, dataset=dataset)
create_shared_avg_table_all_experiments(projects, save_path=table_save_path, dataset=dataset)
metrics_experiment_box_plot(projects, "MASE", save_path=figure_base_path, dataset=dataset)
metrics_experiment_box_plot(projects, "sMAPE", save_path=figure_base_path, dataset=dataset)



# Dataset 2
dataset = "dataset_2"
table_save_path = f"./MastersThesis/tables/results/{dataset}"
if not os.path.isdir(table_save_path):
    os.mkdir(table_save_path)
projects = {
    "sarima": f"{dataset}-sarima",

    "local univariate lstm": f"{dataset}-lstm-local-univariate-tune-400-trials",
    "local multivariate lstm": f"{dataset}-lstm-local-multivariate-tune-400-trials",
    "global univariate lstm": f"{dataset}-lstm-global-univariate-tune-400-trials-exp",
    "global multivariate lstm": f"{dataset}-lstm-global-multivariate-tune-400-trials",

    "local univariate cnn ae lstm": f"{dataset}-cnn-ae-lstm-local-univariate",
    "local multivariate cnn ae lstm": f"{dataset}-cnn-ae-lstm-local-multivariate",
    "global univariate cnn ae lstm": f"{dataset}-cnn-ae-lstm-global-univariate",
    "global multivariate cnn ae lstm": f"{dataset}-cnn-ae-lstm-global-multivariate",
}
create_all_tables_each_experiment(projects, save_path=table_save_path, dataset=dataset)
create_shared_avg_table_all_experiments(projects, save_path=table_save_path, dataset=dataset)
metrics_experiment_box_plot(projects, "MASE", save_path=figure_base_path, dataset=dataset)
metrics_experiment_box_plot(projects, "sMAPE", save_path=figure_base_path, dataset=dataset)





# Dataset 3 (seasonal)
dataset = "dataset_seasonal"
table_save_path = f"./MastersThesis/tables/results/{dataset}"
projects = {
    "sarima": f"{dataset}-sarima",

    "local univariate lstm": "dataset_seasonal-lstm-local-univariate-final",
    "local multivariate lstm": "dataset_seasonal-lstm-local-multivariate-final",
    "global univariate lstm": "dataset_seasonal-lstm-global-univariate-final",
    "global multivariate lstm": "dataset_seasonal-lstm-global-multivariate-final",

    "local univariate cnn ae lstm": f"{dataset}-cnn-ae-lstm-local-univariate",
    "local multivariate cnn ae lstm": f"{dataset}-cnn-ae-lstm-local-multivariate",
    "global univariate cnn ae lstm": f"{dataset}-cnn-ae-lstm-global-univariate",
    "global multivariate cnn ae lstm": f"{dataset}-cnn-ae-lstm-global-multivariate",
}
dataset = "dataset_3"
if not os.path.isdir(table_save_path):
    os.mkdir(table_save_path)
create_all_tables_each_experiment(projects, save_path=table_save_path, dataset=dataset)
create_shared_avg_table_all_experiments(projects, save_path=table_save_path, dataset=dataset)
metrics_experiment_box_plot(projects, "MASE", save_path=figure_base_path, dataset=dataset)
metrics_experiment_box_plot(projects, "sMAPE", save_path=figure_base_path, dataset=dataset)





# Dataset - extra - diff 
dataset = "dataset_diff"
table_save_path = f"./MastersThesis/tables/results/{dataset}"
if not os.path.isdir(table_save_path):
    os.mkdir(table_save_path)
projects = {
    "local univariate lstm dataset 1": "dataset_1-lstm-local-univariate-tune-400-trials",
    "local univariate lstm dataset 3": "dataset_seasonal-lstm-local-univariate-tune-400-trials",
    "local univariate lstm dataset 1 diff": "dataset_1-lstm-local-unviariate-tune-400-trials-with-differencing",
    "local univariate lstm dataset 3 diff": "dataset_seasonal-lstm-local-univariate-differencing-tune-400-trials",
}
create_all_tables_each_experiment(projects, save_path=table_save_path, dataset=dataset)
create_shared_avg_table_all_experiments(projects, save_path=table_save_path, dataset=dataset)




# Variance
dataset = "dataset_variance"
table_save_path = f"./MastersThesis/tables/results/{dataset}"
figure_base_path = f"./MastersThesis/figs/results"
if not os.path.isdir(table_save_path):
    os.mkdir(table_save_path)
variance_experiments = {
    "dataset-high-variance-lstm-local-univariate": "dataset-high-variance-lstm-local-univariate",
    "dataset-high-variance-cnn-ae-lstm-local-univariate": "dataset-high-variance-cnn-ae-lstm-local-univariate",
    "ok-variance-lstm-local-univariate": "dataset-ok-variance-lstm-local-univariate",
    "ok-variance-local-univariate-cnn-ae-lstm": "dataset-ok-variance-cnn-ae-lstm-local-univariate",
    "low-variance-lstm-local-univariate": "dataset-low-variance-lstm-local-univariate",
    "low-variance-cnn-ae-lstm-local-univariate": "dataset-low-variance-cnn-ae-lstm-local-univariate",
}
create_all_tables_each_experiment(variance_experiments, save_path=table_save_path, dataset=dataset)
create_shared_avg_table_all_experiments(variance_experiments, save_path=table_save_path, dataset=dataset)



# Additional box plots
dataset = "dataset_1_diff"
projects = {
    "local univariate lstm dataset 1": "dataset_1-lstm-local-univariate-tune-400-trials",
    "local univariate lstm dataset 1 diff": "dataset_1-lstm-local-unviariate-tune-400-trials-with-differencing",
}
metrics_experiment_box_plot(projects, "MASE", save_path=figure_base_path, dataset=dataset)
metrics_experiment_box_plot(projects, "sMAPE", save_path=figure_base_path, dataset=dataset)

dataset = "dataset_3_diff"
projects = {
    "local univariate lstm seasonal": "dataset_seasonal-lstm-local-univariate-tune-400-trials",
    "local univariate lstm seasonal diff": "dataset_seasonal-lstm-local-univariate-differencing-tune-400-trials",
}
metrics_experiment_box_plot(projects, "MASE", save_path=figure_base_path, dataset=dataset)
metrics_experiment_box_plot(projects, "sMAPE", save_path=figure_base_path, dataset=dataset)


dataset = "dataset_high_variance"
projects = {
    "dataset-high-variance-lstm-local-univariate": "dataset-high-variance-lstm-local-univariate",
    "dataset-high-variance-cnn-ae-lstm-local-univariate": "dataset-high-variance-cnn-ae-lstm-local-univariate",
}
metrics_experiment_box_plot(projects, "MASE", save_path=figure_base_path, dataset=dataset)
metrics_experiment_box_plot(projects, "sMAPE", save_path=figure_base_path, dataset=dataset)

dataset = "dataset_ok_variance"
projects = {
    "ok-variance-lstm-local-univariate": "dataset-ok-variance-lstm-local-univariate",
    "ok-variance-local-univariate-cnn-ae-lstm": "dataset-ok-variance-cnn-ae-lstm-local-univariate",
}
metrics_experiment_box_plot(projects, "MASE", save_path=figure_base_path, dataset=dataset)
metrics_experiment_box_plot(projects, "sMAPE", save_path=figure_base_path, dataset=dataset)

dataset = "dataset_low_variance"
projects = {
    "low-variance-lstm-local-univariate": "dataset-low-variance-lstm-local-univariate",
    "low-variance-cnn-ae-lstm-local-univariate": "dataset-low-variance-cnn-ae-lstm-local-univariate",
}
metrics_experiment_box_plot(projects, "MASE", save_path=figure_base_path, dataset=dataset)
metrics_experiment_box_plot(projects, "sMAPE", save_path=figure_base_path, dataset=dataset)

