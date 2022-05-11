import matplotlib.pyplot as plt
import pandas as pd
import os
import shutil
from typing import List

# TODO: Replace with config value
figure_save_location = "../MScTemplate/figs/code_generated/data_exploration/"


def save_fig_for_raport(fig: plt.Axes, name: str) -> plt.Axes:
    """
    Save figure for raport with given name

    Parameters:
      fig (MplAxes): Figure to save
      name (str): Name of the figure

    Returns:
      MplAxes: Original figure
    """
    plt.figure()
    ax = fig
    plt.rcParams["font.size"] = "16"
    plt.tick_params(labelsize=20)
    ax.set_title(name, fontsize=20)
    return ax.get_figure().savefig(
        f"{figure_save_location}/{name}.png", bbox_inches="tight", facecolor="w", edgecolor="w"
    )


# TODO: Replace with config value
table_save_location = "../MScTemplate/tables/code_generated/data_exploration/"


def dataframe_to_latex_tabular(
    df: pd.DataFrame, caption: str, label: bool, add_index=False, save_local=table_save_location
) -> pd.DataFrame:
    """
    Save dataframe to latex tabular tex file for raport

    Parameters:
      df (pd.Dataframe): Dataframe to save
      caption (str): Caption of the table
      label (bool): If true, add label to table
      add_index (bool): If true, add index to table

    Returns:
      pd.DataFrame: Original dataframe
    """
    table_string = df.to_latex(
        index=add_index,
        bold_rows=True,
        caption=caption,
        label=f"table:{label}",
        header=True,
        multicolumn=True,
        multirow=True,
        multicolumn_format="c",
        # Dont know if this works yet
        position="h",
    )
    table_split = table_string.split("\n")
    table_join = "\n".join(table_split)
    with open(f"{save_local}/{label}.tex", "w") as f:
        f.write(table_join)


# Copy experiment figures from experiments
thesis_figure_location = "./MastersThesis/figs/results"
figure_name = ["Predictions", "Data Prediction"]


def transfer_experiment_figures(experiment: str, figure_types: List[str]):
    # Check if experiment exists and experiment has figures folder
    if not os.path.isdir(f"./models/{experiment}") or not os.path.isdir(
        f"./models/{experiment}/figures"
    ):
        raise NotADirectoryError(
            f"The experiment {experiment} does not exist or have a figure directory."
        )
    # Check that the folder to store files exists
    if not os.path.isdir(thesis_figure_location):
        raise NotADirectoryError(
            f"The directory for storing result images does not exist. Create a driectory at: {thesis_figure_location}"
        )
    if not os.path.isfile(f"{thesis_figure_location}/{experiment}"):
        os.mkdir(f"{thesis_figure_location}/{experiment}")

    # Figure names are [Predictions, Training data, Training data approximation, Data Prediction
    experiment_figures = os.listdir(f"./models/{experiment}/figures")
    for image_name in experiment_figures:
        if any([x in image_name for x in figure_types]):
            # Copy file from src to destination
            shutil.copyfile(
                f"./models/{experiment}/figures/{image_name}",
                f"{thesis_figure_location}/{experiment}/{image_name}",
            )
