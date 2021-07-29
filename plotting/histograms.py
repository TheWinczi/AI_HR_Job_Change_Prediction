import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA


def draw_counts_histogram(data: np.ndarray,
                          title: str = None,
                          xlabel: str = None,
                          ylabel: str = None):
    sums = []
    unique_ticks = np.unique(data)
    for tic in unique_ticks:
        sums.append((data == tic).sum())

    length = len(sums)
    colors = list(zip(np.random.rand(length), np.random.rand(length), np.random.rand(length), [1 for _ in sums]))
    plt.bar(np.arange(length), sums, color=colors)
    plt.xticks(np.arange(length), unique_ticks)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()


def draw_correlation_histogram(x: np.ndarray,
                               y: np.ndarray,
                               title: str = None,
                               xlabel: str = None,
                               ylabel: str = None):
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

        if i == 0:
            sorted_indices = np.argsort(bars_values[i])

        plotting_bars_values = np.array(bars_values[i])[sorted_indices]
        plt.bar(np.arange(len(ticks)), plotting_bars_values, bottom=bottom_values)
        bottom_values = plotting_bars_values

    plt.xticks(np.arange(len(ticks)), ticks)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()


def plot_LDA_features_importances(x: np.ndarray,
                                  y: np.ndarray,
                                  title: str = None,
                                  xlabel: str = None,
                                  ylabel: str = None):
    """
    Plot features importances of input array using PCA
    dimensions reduction algorithm.

    Parameters
    ----------
    x : ndarray
        Array of lists/tuples of input features in shape
        [(f_00, f_01, f_02, ...), (f_10, f_11, f_12, ...) ...]

    y : ndarray
        Array of labels belong to each lists/tuples in input x array.

    title : str {default: None}
        Plot title. Could be None.

    xlabel : str {default: None}
        Label of x axis of plot. Could be None.

    ylabel : str {default: None}
        Label of y axis of plot. Could be None.
    """
    lda = LDA(n_components=2)
    lda.fit(x, y)

    # pca = PCA(n_components=None)
    # pca.fit(x)

    features_importances = lda.explained_variance_ratio_

    plt.figure(figsize=(16, 9))
    plt.bar(np.arange(len(features_importances)), features_importances)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()
    plt.show()
