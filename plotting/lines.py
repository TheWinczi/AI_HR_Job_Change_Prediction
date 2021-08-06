import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from plotting import _get_lines_styles


def plot_roc_line(estimators: list,
                  x: np.ndarray,
                  y: np.ndarray,
                  labels: list[str] = None,
                  colors: list = None,
                  lines_styles: list[str] = None,
                  title: str = None,
                  xlabel: str = None,
                  ylabel: str = None):
    """
    Draw ROC lines of estimators on one plot.

    Parameters
    ----------
    estimators : list
        List of trained estimators ready to predict probability of labels.

    x : ndarray
        Array of values of each feature importance in shape
        [f_0, f_1, f_2, ... ] which sum is equal 1.

    y : ndarray
        List of target values which belongs to each
        tuple/list of input x array.

    labels : list[str] {default: None}
        List of labels which are used as lines legends. Could be None.

    colors : list {default: None}
        List of lines colors. Could be None.

    lines_styles : list[str] {default: None}
        List of lines drawing styles. Could be None.

    title : str {default: None}
        Chart title. Could be None.

    xlabel : str {default: None}
        Label of x axis of chart. Could be None.

    ylabel : str {default: None}
        Label of y axis of chart. Could be None.
    """

    if lines_styles is None:
        lines_styles = _get_lines_styles(len(estimators))

    plt.figure()

    for i, est in enumerate(estimators):
        label = None if labels is None else labels[i]
        color = None if colors is None else colors[i]
        line_style = None if lines_styles is None else lines_styles[i]

        _draw_roc_line(est,
                       x, y,
                       label,
                       color,
                       line_style)

    _draw_random_guessing_roc_line()
    _draw_perfect_roc_line()

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()


def _draw_roc_line(clf,
                   x: np.ndarray,
                   y_true: np.ndarray,
                   label: str = None,
                   color: list[float, float, float] = None,
                   line_style: str = '-'):
    """
    Draw roc line on created plot.

    Parameters
    ----------
    clf
        Trained classifier ready to predict.

    x : ndarray
        Array of values of each feature importance in shape
        [f_0, f_1, f_2, ... ] which sum is equal 1.

    y_true : ndarray
        Array of true labels belongs to input x data.

    color
        Color of drawing line.

    line_style : str {default: '-'}
        Style of drawing line.
    """
    scores = clf.predict_proba(x)
    scores = list(map(lambda item: item[1], scores))

    fpr, tpr, _ = roc_curve(y_true,
                            scores,
                            pos_label=1)

    auc_ = auc(fpr, tpr)
    label = f"area = {auc}" if label is None else f"{label} (area = {auc_})"

    plt.plot(fpr, tpr,
             linestyle=line_style,
             color=color,
             label=label,
             lw=1)


def _draw_random_guessing_roc_line(color: str = "black",
                                   line_style: str = '--'):
    plt.plot([0, 1],
             [0, 1],
             linestyle=line_style,
             color=color,
             label="Random Guessing")


def _draw_perfect_roc_line(color: str = "black",
                           line_style: str = ':'):
    plt.plot([0, 0, 1],
             [0, 1, 1],
             linestyle=line_style,
             color=color,
             label="Perfect Performance")
