import numpy as np
from .k_nearest_neighbors import k_nearest_neighbors
from .decision_tree import decision_tree
from .random_forest import random_forest
from .support_vectors import support_vectors
from .logistic_regression import logistic_regression
from .team import team


def try_all_classifiers(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray = None, y_test: np.ndarray = None,
                        compare: bool = True):
    """
    Try all classifiers which was implemented to classify input data.

    Parameters
    ----------
    X_train : ndarray
        Array of data (points) as a list of tuples/lists
        in shape [(x_0, y_0), (x_1, y_1) ... ].

    y_train : ndarray
        Array of labels belongs to X_train data.

    X_test : ndarray {default: None}
        Array of data (points) as a list of tuples/lists
        in shape [(x_0, y_0), (x_1, y_1) ... ]. Could be None.

    y_test : ndarray {default: None}
        Array of labels belongs to X_test data. Could be None.

    compare : bool {default: True}
        Are classifiers needed to be compared on some plots

    Returns
    -------
    clf
        The best trained classifier ready to predict.
    """
    # dnn_ = deep_neural_network(X_train, y_train, X_test, y_test)

    # team_ = team(X_train, y_train, X_test, y_test)

    knn_ = k_nearest_neighbors(X_train, y_train, X_test, y_test)

    tree_ = decision_tree(X_train, y_train, X_test, y_test)

    forest_ = random_forest(X_train, y_train, X_test, y_test)

    svm_ = support_vectors(X_train, y_train, X_test, y_test)

    log_reg_ = logistic_regression(X_train, y_train, X_test, y_test)

    return knn_
