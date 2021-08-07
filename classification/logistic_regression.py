import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def logistic_regression(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray = None, y_test: np.ndarray = None):
    """
    Logistic Regression classification algorithm used for classification input data.

    References
    ----------
    [1] https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

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
    logistic
        Trained classifier ready to predict.
    """
    # _check_logistic_regression_params(X_train, y_train)

    logistic = LogisticRegression(C=0.01,
                                  solver='newton-cg',
                                  tol=10**(-3),
                                  random_state=1)
    logistic.fit(X_train, y_train)

    if X_test is not None and y_test is not None and len(X_test) == len(y_test):
        y_pred = logistic.predict(X_test)
        print(f"Logistic Regression test accuracy: {accuracy_score(y_test, y_pred)}")

    return logistic


def _check_logistic_regression_params(X: np.ndarray, y: np.ndarray):
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
    Cs = [0.001, 0.01, 0.1, 1, 10, 20, 40, 50]
    solvers = ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
    param_grid = {
        "C": Cs,
        "solver": solvers,
    }

    gs = GridSearchCV(estimator=LogisticRegression(random_state=1, multi_class='ovr', tol=10**(-3)),
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=10)
    gs.fit(X, y)
    print(f"The best combination: {gs.best_params_}, score: {gs.best_score_}")
