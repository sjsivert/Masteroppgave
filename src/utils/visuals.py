# TODO: One visualize with one list input
# TODO: One with multiple list inputs
# TODO: One with an update method with single and multiple inputs
# TODO: Each IModel implementation can create their own methods for what they wish to visualize.
from typing import List

import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def visualize_data_series(
    title: str,
    data_series: List,
    data_labels: List[str],
    colors: List[str] = ["blue", "red", "orange"],
    x_label: str = "x",
    y_label: str = "y",
) -> Figure:
    fig = plt.figure(num=title)
    plt.clf()
    plt.title(f"{title}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    for idx, series in enumerate(data_series):
        plt.plot(series, label=data_labels[idx], color=colors[idx] if len(colors) > idx else "blue")
    plt.close()
    return fig
