import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def decision_tree(X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray = None, y_test: np.ndarray = None):
    """
    Decision Tree classification algorithm used for classification input data.

    References
    --------
    [1] https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

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
    tree
        Trained classifier ready to predict.
    """
    # _check_tree_params(X_train, y_train)

    tree = DecisionTreeClassifier(max_depth=8,
                                  criterion='entropy',
                                  random_state=1)
    tree.fit(X_train, y_train)

    if X_test is not None and y_test is not None and len(X_test) == len(y_test):
        y_pred = tree.predict(X_test)
        print(f"Decision Tree test accuracy: {accuracy_score(y_test, y_pred)}")

    return tree


def _check_tree_params(X: np.ndarray, y: np.ndarray):
    """
    Check all parameters needed in Decision Tree classification algorithm.
    Show results on plots.

    Parameters
    ----------
    X : ndarray
        Array of input data (points) as a list o tuples/lists
        in shape [(x_0, y_0), (x_1, y_1) ... ].

    y : ndarray
        Array of labels belongs to input X_train data.
    """
    max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    criterions = ['gini', 'entropy']
    param_grid = {
        "max_depth": max_depths,
        "criterion": criterions
    }

    gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=1),
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=10)
    gs.fit(X, y)
    print(f"The best combination: {gs.best_params_}, score: {gs.best_score_}")
