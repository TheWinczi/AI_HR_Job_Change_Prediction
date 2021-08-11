import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
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

    plt.figure(figsize=(16, 9))

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


def plot_learning_history(history: dict[str, list]):
    """
    Plot learning history in one plot.
    Parameters
    ----------
    history : dict
        Dictionary object stores history of learning e.g.
        history = {'accuracy' : [....],
                   'loss': [...]}
    """
    num_epochs = len(history['accuracy'])

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(range(1, num_epochs+1), history['loss'], label="train loss")

    if 'val_loss' in history.keys():
        ax1.plot(range(1, num_epochs + 1), history['val_loss'], 'k--', label="validation loss")

    ax1.set_title('Loss Function')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('value')
    ax1.legend()

    ax2.plot(range(1, num_epochs+1), history['accuracy'], label="train accuracy")

    if 'val_accuracy' in history.keys():
        ax2.plot(range(1, num_epochs + 1), history['val_accuracy'], 'k--', label="validation accuracy")

    ax2.set_title('Learning Accuracy')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('value')
    ax2.legend()

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
    if isinstance(clf, tf.keras.Sequential):
        scores = clf.predict(x)
        scores = list(map(lambda item: item[0], scores))
    else:
        scores = clf.predict_proba(x)
        scores = list(map(lambda item: item[1], scores))

    fpr, tpr, _ = roc_curve(y_true,
                            scores,
                            pos_label=1)

    auc_ = auc(fpr, tpr)
    label = f"area = {auc}" if label is None else f"{label} (area = {round(auc_, 4)})"

    plt.plot(fpr, tpr,
             linestyle=line_style,
             color=color,
             label=label,
             lw=3,
             alpha=0.5)


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
