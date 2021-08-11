import numpy as np
import pandas as pd
from .deep_neural_network import deep_neural_network
from .k_nearest_neighbors import k_nearest_neighbors
from .decision_tree import decision_tree
from .random_forest import random_forest
from .support_vectors import support_vectors
from .logistic_regression import logistic_regression
from .team import team
from plotting import *


def try_all_classifiers(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray = None, y_test: np.ndarray = None,
                        df: pd.DataFrame = None,
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

    df : DataFrame
        DataFrame storing all data to train and test. Could be None.
        Only used in Deep Neural Network, otherwise ignored.

    compare : bool {default: True}
        Are classifiers needed to be compared on some plots

    Returns
    -------
    clf
        The best trained classifier ready to predict.
    """
    dnn_ = deep_neural_network(X_train, y_train, X_test, y_test)

    tree_ = decision_tree(X_train, y_train, X_test, y_test)

    knn_ = k_nearest_neighbors(X_train, y_train, X_test, y_test)

    forest_ = random_forest(X_train, y_train, X_test, y_test)

    log_reg_ = logistic_regression(X_train, y_train, X_test, y_test)

    svm_ = support_vectors(X_train, y_train, X_test, y_test)

    team_ = team(X_train, y_train, X_test, y_test)

    if compare:
        estimators = [dnn_, knn_, tree_, log_reg_, forest_, team_, svm_]
        plot_roc_line(estimators,
                      X_test, y_test,
                      labels=["DNN", "KNN", "Tree", "Log Reg", "Forest", "Team", "SVM"],
                      title="ROC classifiers comparison (no reduction)",
                      xlabel="false positives",
                      ylabel="true positives")

        best_est = [dnn_, log_reg_, svm_]
        best_est_titles = ["DNN", "Logistic Regression", "SVM"]
        for i, est in enumerate(best_est):
            plot_confusion_matrix(est,
                                  X_test,
                                  y_test,
                                  best_est_titles[i])

    return None
