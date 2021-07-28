import numpy as np
import matplotlib.pyplot as plt
from .utilities import _get_colors


def plot_points(x: np.ndarray,
                y: np.ndarray,
                title: str = None,
                labels: list[str] = None,
                xlabel: str = None,
                ylabel: str = None):
    """
    Plot points on one figure. Points could be plot only on
    2D euclidean surface.

    Parameters
    ----------
    x : ndarray
        Array of lists/tuples of points coordinates in shape
        [(x_0, y_0), (x_1, y_1), ...].

    y : ndarray
        Labels belongs to each point in input x array. It is used
        for distinguish between points - different label is equal
        to different marker/color. Could be array of int, bool, str.

    title : str {default: None}
        Title of plot. Could be None.

    labels : list[str] {default: None}
        List of labels needed to plot legend - each y list value has
        different label. If y is None - labels is ignored.

    xlabel : str {default: None}
        Label of x axis of plot.

    ylabel : str {default: None}
        Label of y axis of plot.
    """

    plt.figure(figsize=(16, 9))

    if y is not None and len(x) == len(y):
        unique_y_values = np.unique(y)
        colors = _get_colors(len(unique_y_values), kind="str")

        for i, y_value in enumerate(unique_y_values):
            label = labels[i] if labels is not None else None
            color = colors[i]

            indices = y == y_value
            x_coords, y_coords = x[indices, 0], x[indices, 1]
            plt.scatter(x_coords, y_coords,
                        c=color,
                        label=label)
    else:
        x_coords, y_coords = x[:, 0], x[:, 1]
        plt.scatter(x_coords, y_coords)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.legend()
    plt.tight_layout()
    plt.show()



