import typing

from matplotlib.figure import Figure
from plotly.basedatatypes import BaseFigure
from plotly.graph_objs import Figure as PlotlyFigure


def combine_subfigure_titles(figure: Figure) -> str:
    if type(figure) is PlotlyFigure:
        return figure.layout.title.text
    else:
        titles = list(map(lambda ax: ax.title.get_text(), figure.axes))
        return ", ".join(titles)
