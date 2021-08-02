import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def k_nearest_neighbors(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray = None, y_test: np.ndarray = None):
    """
    K-Nearest Neighbours classification algorithm used for classification input data.

    References
    ----------
    [1] https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

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
    knn
        Trained classifier ready to predict.
    """
    # _check_knn_params(X_train, y_train)

    knn = KNeighborsClassifier(n_neighbors=15, metric='manhattan')
    knn.fit(X_train, y_train)

    if X_test is not None and y_test is not None and len(X_test) == len(y_test):
        y_pred = knn.predict(X_test)
        print(f"KNN test accuracy: {accuracy_score(y_test, y_pred)}")

    return knn


def _check_knn_params(X: np.ndarray, y: np.ndarray):
    """
    Check all parameters needed in K-Nearest Neighbours (KNN) classification algorithm.
    Show results on plots.

    Parameters
    ----------
    X : ndarray
        Array of input data (points) as a list of tuples/lists
        in shape [(x_0, y_0), (x_1, y_1) ... ].

    y : ndarray
        Array of labels belongs to input X data.
    """
    n_neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    metrics = ['minkowski', 'manhattan']
    param_grid = {
        "n_neighbors": n_neighbors,
        "metric": metrics
    }

    gs = GridSearchCV(estimator=KNeighborsClassifier(),
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=10)
    gs.fit(X, y)
    print(f"The best combination: {gs.best_params_}, score: {gs.best_score_}")
