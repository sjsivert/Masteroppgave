from matplotlib.figure import Figure


def _combine_subfigure_titles(figure: Figure) -> str:
    titles = list(map(lambda ax: ax.title.get_text(), figure.axes))
    return ", ".join(titles)
