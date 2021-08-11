import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(clf,
                          x: np.ndarray,
                          y_true: np.ndarray,
                          title: str = None):
    """
    Plot confusion matrix on one figure.

    Parameters
    ----------
    clf
        Trained classifier ready to predict.

    x : ndarray
        Array of input data (points) as a list of tuples/lists
        in shape [(x_0, y_0), (x_1, y_1) ... ].

    y_true : ndarray
        Array of true labels belongs to input x data.

    title : str {default: None}
        Title of plot. Program will add "Confusion Matrix of" at the beginning of given title.
    """
    y_pred = clf.predict(x)

    if isinstance(clf, tf.keras.Sequential):
        y_pred = list(map(lambda item: 0 if item[0] <= 0.5 else 1, y_pred))

    confmat = confusion_matrix(y_true, y_pred)

    plt.matshow(confmat, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            plt.text(x=j, y=i,
                     s=confmat[i, j],
                     va='center', ha='center')

    plt.xlabel("Predicted Label")
    plt.ylabel("Real Label")
    plt.title(f"Confusion Matrix of {title}")
    plt.show()
