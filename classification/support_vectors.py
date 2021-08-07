import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def support_vectors(X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray = None, y_test: np.ndarray = None):
    """
    C-Support Vector Classification classification algorithm used for classification input data.

    References
    ----------
    [1] https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

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
    svm
        Trained classifier ready to predict.
    """
    # _check_svm_params(X_train, y_train)

    svm = SVC(C=1,
              kernel='rbf',
              probability=True,
              random_state=1)
    svm.fit(X_train, y_train)

    if X_test is not None and y_test is not None and len(X_test) == len(y_test):
        y_pred = svm.predict(X_test)
        print(f"SVM test accuracy: {accuracy_score(y_test, y_pred)}")

    return svm


def _check_svm_params(X: np.ndarray, y: np.ndarray):
    """
    Check all parameters needed in SVM classification algorithm.
    Show results on plots.

    Parameters
    ----------
    X : ndarray
        Array of input data (points) as a list of tuples/lists
        in shape [(x_0, y_0), (x_1, y_1) ... ].

    y : ndarray
        Array of labels belongs to input X data.
    """
    Cs = [0.01, 0.1, 1, 2, 3, 4, 5, 10]
    kernels = ["rbf"] # ["linear", "poly", "rbf", "sigmoid"]
    param_grid = {
        "C": Cs,
        "kernel": kernels
    }

    gs = GridSearchCV(estimator=SVC(random_state=1),
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=10)
    gs.fit(X, y)
    print(f"The best combination: {gs.best_params_}, score: {gs.best_score_}")
