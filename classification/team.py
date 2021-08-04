import numpy as np
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


def team(X_train: np.ndarray, y_train: np.ndarray,
         X_test: np.ndarray = None, y_test: np.ndarray = None):
    """
    Classification algorithm using team of classifiers used for classification input data.

    References
    ----------
    [1] https://scikit-learn.org/stable/modules/ensemble.html

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
    team
        Trained classifier ready to predict.
    """
    # _check_team_params(X_train, y_train)

    knn_ = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
    svm_ = SVC(C=60, kernel='rbf', random_state=1)
    tree_ = DecisionTreeClassifier(random_state=1)

    voting = VotingClassifier([('knn', knn_),
                               ('svm', svm_),
                               ('tree', tree_)],
                              n_jobs=-1)
    voting.fit(X_train, y_train)

    if X_test is not None and y_test is not None and len(X_test) == len(y_test):
        y_pred = voting.predict(X_test)
        print(f"TEAM VOTING test accuracy: {accuracy_score(y_test, y_pred)}")

    return voting


def _check_team_params(X: np.ndarray, y: np.ndarray):
    """
    Check all parameters needed in Team Classification algorithm.
    Show results on plots.

    Parameters
    ----------
    X : ndarray
        Array of input data (points) as a list of tuples/lists
        in shape [(x_0, y_0), (x_1, y_1) ... ].

    y : ndarray
        Array of labels belongs to input X data.
    """
    tree = DecisionTreeClassifier(criterion='entropy', random_state=1)

    for i in range(100, 301, 50):
        bagging = BaggingClassifier(base_estimator=tree,
                                    n_estimators=i,
                                    max_samples=1.0,
                                    max_features=1.0,
                                    random_state=1)
        bagging.fit(X, y)
        print(bagging.score(X, y))

    knn_ = KNeighborsClassifier(n_neighbors=1, metric='manhattan')
    svm_ = SVC(C=60, kernel='rbf', random_state=1)
    tree_ = DecisionTreeClassifier(random_state=1)

    voting = VotingClassifier([('knn', knn_), ('svm', svm_), ('tree', tree_)],
                              n_jobs=-1)
    voting.fit(X, y)
