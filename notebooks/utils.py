import matplotlib.pyplot as plt
#from qbstyles import mpl_style
import pandas as pd

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
  #mpl_style()
  #plt.style.use('seaborn-poster')
  plt.figure()
  ax = fig
  plt.rcParams['font.size'] = '20'
  plt.tick_params(labelsize=20)
  ax.set_title(name, fontsize=20)
  return ax.get_figure().savefig(f"{figure_save_location}/{name}.png", bbox_inches='tight', facecolor='w', edgecolor='w')


table_save_location = "../MScTemplate/tables/code_generated/data_exploration/"
def dataframe_to_latex_tabular(df: pd.DataFrame, caption: str, label: bool, add_index=False) -> pd.DataFrame:
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
        position="htbp",
        )
    table_split = table_string.split("\n")
    table_join = "\n".join(table_split)
    with open(f"{table_save_location}/{label}.tex", "w") as f:
          f.write(table_join)