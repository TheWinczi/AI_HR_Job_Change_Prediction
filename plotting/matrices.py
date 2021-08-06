import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(clf,
                          x: np.ndarray,
                          y_true: np.ndarray):
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
    """
    y_pred = clf.predict(x)
    confmat = confusion_matrix(y_true, y_pred)

    plt.matshow(confmat, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            plt.text(x=j, y=i,
                     s=confmat[i, j],
                     va='center', ha='center')

    plt.xlabel("Predicted Label")
    plt.ylabel("Real Label")
    plt.show()
