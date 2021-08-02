import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def random_forest(X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray = None, y_test: np.ndarray = None):
    """
    Random Forest classification algorithm used for classification input data.

    References
    ----------
    [1] https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

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
    forest
        Trained classifier ready to predict.
    """
    # _check_forest_params(X_train, y_train)

    forest = RandomForestClassifier(n_estimators=130,
                                    max_depth=10,
                                    criterion='entropy',
                                    random_state=1)
    forest.fit(X_train, y_train)

    if X_test is not None and y_test is not None and len(X_test) == len(y_test):
        y_pred = forest.predict(X_test)
        print(f"Random Forest test accuracy: {accuracy_score(y_test, y_pred)}")

    return forest


def _check_forest_params(X: np.ndarray, y: np.ndarray):
    """
    Check all parameters needed in Random Forest classification algorithm.
    Show results on plots.

    Parameters
    ----------
    X : ndarray
        Array of input data (points) as a list of tuples/lists
        in shape [(x_0, y_0), (x_1, y_1) ... ].

    y : ndarray
        Array of labels belongs to input X data.
    """
    max_depths = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    n_estimators = [120, 130, 140, 150, 160, 170, 180]
    criterions = ['gini', 'entropy']
    param_grid = {
        "n_estimators": n_estimators,
        "max_depth": max_depths,
        "criterion": criterions
    }
    gs = GridSearchCV(estimator=RandomForestClassifier(random_state=1, n_jobs=-1),
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=10,
                      n_jobs=-1)
    gs.fit(X, y)
    print(f"The best combination: {gs.best_params_}, score: {gs.best_score_}")
