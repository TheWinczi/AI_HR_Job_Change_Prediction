import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from plotting import *


def plot_data_columns_counts(x: np.ndarray,
                             titles: list[str] = None,
                             xlabels: list[str] = None,
                             ylabels: list[str] = None,
                             suptitle: str = None,
                             sort: bool = False):
    """
    Plot data columns values counts.

    Parameters
    ----------
    x : ndarray
        Array of lists/tuples of data features in shape
        [(f_00, f_01, f_02, ...), (f_10, f_11, f_12, ...) ...]

    titles : list[str] {default: None}
        List of each chart title. Could be None.

    xlabels : list[str] {default: None}
        List of each label of x axis of chart. Could be None.

    ylabels : list[str] {default: None}
        List of each label of y axis of chart. Could be None.

    suptitle : str {default: None}
        Suptitle, main title of whole plot.

    sort : bool {default: False}
        Are columns counts have to be sorted in ascending order.
    """
    cols_values = []
    for i in range(len(x[0])):
        cols_values.append(x[:, i])

    fig_rows, fig_cols = calculate_figure_dims(len(cols_values))

    plt.figure(figsize=(16, 9))
    for i, values in enumerate(cols_values):
        plt.subplot(fig_rows, fig_cols, i+1)

        title = None if titles is None else titles[i]
        xlabel = None if xlabels is None else xlabels[i]
        ylabel = None if ylabels is None else ylabels[i]

        draw_data_column_counts(values,
                                title=title,
                                xlabel=xlabel,
                                ylabel=ylabel,
                                sort=sort)
    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


def draw_data_column_counts(x: np.ndarray,
                            title: str = None,
                            xlabel: str = None,
                            ylabel: str = None,
                            sort: bool = False):
    """
    Draw data columns values counts on created plot.

    Parameters
    ----------
    x : ndarray
        Array of values of each feature importance in shape
        [f_0, f_1, f_2, ... ] which sum is equal 1.

    y : ndarray
        List of target values which belongs to each
        tuple/list of input x array.

    title : str {default: None}
        Chart title. Could be None.

    xlabel : str {default: None}
        Label of x axis of chart. Could be None.

    ylabel : str {default: None}
        Label of y axis of chart. Could be None.

    sort: bool {default: False}
        Are columns counts have to be sorted in ascending order.
    """
    bars_values = []
    unique_ticks = np.unique(x)
    for tic in unique_ticks:
        bars_values.append((x == tic).sum())

    length = len(bars_values)
    colors = list(zip(np.random.rand(length),       # R
                      np.random.rand(length),       # G
                      np.random.rand(length),       # B
                      [1 for _ in range(length)]))  # A

    if sort:
        indices = np.argsort(bars_values)
    else:
        indices = np.arange(length)

    plotting_bars_values = np.array(bars_values)[indices]

    plt.bar(np.arange(length), plotting_bars_values, color=colors)
    plt.xticks(np.arange(length), unique_ticks)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()


def plot_data_target_dependencies(x: np.ndarray,
                                  y: np.ndarray,
                                  titles: list[str] = None,
                                  xlabels: list[str] = None,
                                  ylabels: list[str] = None,
                                  suptitle: str = None,
                                  sort: bool = False):
    """
    Analyze data dependencies on target. Show is any dependencies
    between specific columns values and target.

    Parameters
    ----------
    x : ndarray
        Array of lists/tuples of data features in shape
        [(f_00, f_01, f_02, ...), (f_10, f_11, f_12, ...) ...]

    titles : list[str] {default: None}
        List of each chart title. Could be None.

    xlabels : list[str] {default: None}
        List of each label of x axis of chart. Could be None.

    ylabels : list[str] {default: None}
        List of each label of y axis of chart. Could be None.

    suptitle : str {default: None}
        Suptitle, main title of whole plot.

    sort : bool {default: False}
        Are dependencies have to be sorted.
    """
    cols_values = []
    for i in range(len(x[0])):
        cols_values.append(x[:, i])

    fig_rows, fig_cols = calculate_figure_dims(len(cols_values))

    plt.figure(figsize=(16, 9))
    for i, values in enumerate(cols_values):
        plt.subplot(fig_rows, fig_cols, i + 1)

        title = None if titles is None else titles[i]
        xlabel = None if xlabels is None else xlabels[i]
        ylabel = None if ylabels is None else ylabels[i]

        draw_data_target_dependencies(values, y,
                                      title=title,
                                      xlabel=xlabel,
                                      ylabel=ylabel,
                                      sort=sort)
    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()


def draw_data_target_dependencies(x: np.ndarray,
                                  y: np.ndarray,
                                  title: str = None,
                                  xlabel: str = None,
                                  ylabel: str = None,
                                  sort: bool = False):
    """
    Draw data dependency on target on created plot.

    Parameters
    ----------
    x : ndarray
        Array of values of each feature importance in shape
        [f_0, f_1, f_2, ... ] which sum is equal 1.

    y : ndarray
        List of target values which belongs to each
        tuple/list of input x array.

    title : str {default: None}
        Chart title. Could be None.

    xlabel : str {default: None}
        Label of x axis of chart. Could be None.

    ylabel : str {default: None}
        Label of y axis of chart. Could be None.

    sort: bool {default: False}
        Are dependencies have to be sorted.
    """
    targets = np.unique(y)
    ticks = np.unique(x)

    bars_values = []
    targets_counts = [0 for _ in ticks]

    for target in targets:
        sums = []
        for i, tick in enumerate(ticks):
            indices = np.logical_and(x == tick, y == target)
            indices_sum = sum(indices)
            sums.append(indices_sum)
            targets_counts[i] += indices_sum
        bars_values.append(sums.copy())

    bottom_values = [0 for _ in range(len(ticks))]
    sorted_indices = range(len(targets_counts))
    for i in range(targets.shape[0]):
        for j, count in enumerate(targets_counts):
            bars_values[i][j] = bars_values[i][j] / count

        if sort and i == 0:
            sorted_indices = np.argsort(bars_values[i])

        plotting_bars_values = np.array(bars_values[i])[sorted_indices]
        plt.bar(np.arange(len(ticks)), plotting_bars_values, bottom=bottom_values)
        bottom_values = plotting_bars_values

    plt.xticks(np.arange(len(ticks)), ticks)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()


def plot_PCA_features_importances(x: np.ndarray,
                                  title: str = None,
                                  xlabel: str = None,
                                  ylabel: str = None):
    """
    Plot features importances of input array using
    dimensions reduction algorithm (PCA).

    Parameters
    ----------
    x : ndarray
        Array of lists/tuples of input features in shape
        [(f_00, f_01, f_02, ...), (f_10, f_11, f_12, ...) ...]

    title : str {default: None}
        Chart title. Could be None.

    xlabel : str {default: None}
        Label of x axis of chart. Could be None.

    ylabel : str {default: None}
        Label of y axis of chart. Could be None.
    """
    pca = PCA(n_components=None)
    pca.fit(x)

    plt.figure(figsize=(16, 9))

    features_importances = pca.explained_variance_ratio_

    _draw_features_importances(features_importances,
                               title=title,
                               xlabel=xlabel,
                               ylabel=ylabel)
    plt.tight_layout()
    plt.show()


def _draw_features_importances(features_importances: np.ndarray,
                               bars_ticks: list[str] = None,
                               title: str = None,
                               xlabel: str = None,
                               ylabel: str = None):
    """
    Draw features importances on created plot.

    Parameters
    ----------
    features_importances : ndarray
        Array of values of each feature importance in shape
        [f_0, f_1, f_2, ... ] which sum is equal 1.

    bars_ticks : list[str] {default: None}
        List of bars ticks which are used as bars labels on
        bars chart.

    title : str {default: None}
        Chart title. Could be None.

    xlabel : str {default: None}
        Label of x axis of chart. Could be None.

    ylabel : str {default: None}
        Label of y axis of chart. Could be None.
    """
    features_count = len(features_importances)
    x = np.arange(features_count)
    steps_values = [0 for i in range(features_count)]

    for i, importance in enumerate(features_importances):
        steps_values[i] = importance + steps_values[i - 1]

    plt.bar(x, features_importances, align='center', label="Single variance of features")
    plt.step(x, steps_values, c="black", where="mid", label="Total variance of features")
    plt.xticks(bars_ticks)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.tight_layout()
