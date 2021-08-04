import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score


# TODO
def deep_neural_network(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray = None, y_test: np.ndarray = None):
    """
    Deep Neural Network created for classification input data.

    References
    ----------
    [1] https://keras.io/

    Parameters
    ----------
    X_train : ndarray
        Array of input data (points) as a list of tuples/lists
        in shape [(x_0, y_0), (x_1, y_1) ... ].

    y_train : ndarray
        Array of labels belongs to input X_train data.

    X_test : ndarray {default: None}
        Array of data (points) as a list of tuples/lists
        in shape [(x_0, y_0), (x_1, y_1) ... ]. Could be None.

    y_test : ndarray {default: None}
        Array of labels belongs to X_test data. Could be None.

    Returns
    -------
    dnn
        Trained classifier ready to predict.
    """
    _check_deep_network_params(X_train, y_train)
    pass


# TODO
def _check_deep_network_params(X: np.ndarray, y: np.ndarray):
    """
    Check the number of layers, types of layers, activation functions etc.
    needed in a deep neural network. Show results on plots.

    Parameters
    ----------
    X : ndarray
        Array of input data (points) as a list of tuples/lists
        in shape [(x_0, y_0), (x_1, y_1) ... ].

    y : ndarray
        Array of labels belongs to input X data.
    """
    pass
