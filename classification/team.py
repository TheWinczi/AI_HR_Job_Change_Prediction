import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


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

    log_ = LogisticRegression(C=0.0001,
                              solver="liblinear",
                              tol=10 ** (-3),
                              random_state=1)

    forest_ = RandomForestClassifier(n_estimators=100,
                                     max_depth=3,
                                     criterion='gini',
                                     random_state=1)

    tree_ = DecisionTreeClassifier(max_depth=2,
                                   criterion='entropy',
                                   random_state=1)

    voting = VotingClassifier([('logistic', log_),
                               ('forest', forest_),
                               ('tree', tree_)],
                              voting='soft')
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
    pass
