import numpy as np
import matplotlib.pyplot as plt


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
    for i in range(len(targets)):
        for j, count in enumerate(targets_counts):
            bars_values[i][j] = bars_values[i][j] / count

        # bars_sorted = np.sort(list(zip(np.arange(len(ticks)), bars_values[i])), axis=1)
        # indexes, bars = zip(*bars_sorted)

        plt.bar(np.arange(len(ticks)), bars_values[i], bottom=bottom_values)
        bottom_values = bars_values[i]

    plt.xticks(np.arange(len(ticks)), ticks)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
