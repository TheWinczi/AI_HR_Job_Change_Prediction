from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# --------------------------------------------------------
# DECISION TREE
def tree_classification(X_train, y_train, X_test, y_test):
    # check_tree_parameters(X_train, y_train)
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=1)
    tree.fit(X_train, y_train)
    print(f"(TREE) Train accuracy {tree.score(X_train, y_train)}\nTest accuracy {tree.score(X_test, y_test)}")


def check_tree_parameters(X, y):
    depth_range = [2, 3, 4, 5, 6, 7, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    criterions = ['gini', 'entropy']
    param_grid = {'criterion': criterions,
                  'max_depth': depth_range}
    gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=1),
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=10,
                      n_jobs=-1)
    gs = gs.fit(X, y)
    print(gs.best_score_)
    print(gs.best_params_)


# --------------------------------------------------------
# RANDOM FOREST
def random_forest_classification(X_train, y_train, X_test, y_test):
    # check_random_forest_parameters(X_train, y_train)
    forest = RandomForestClassifier(criterion='entropy', n_estimators=130, max_depth=10, random_state=1, n_jobs=-1)
    forest.fit(X_train, y_train)
    print(f"(FOREST) Train accuracy {forest.score(X_train, y_train)}\nTest accuracy {forest.score(X_test, y_test)}")


def check_random_forest_parameters(X, y):
    depth_range = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    criterions = ['gini', 'entropy']
    estimators_range = [100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
    param_grid = {'criterion': criterions,
                  'max_depth': depth_range,
                  'n_estimators': estimators_range}
    gs = GridSearchCV(estimator=RandomForestClassifier(random_state=1, n_jobs=-1),
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=10,
                      n_jobs=-1)
    gs.fit(X, y)
    print(gs.best_score_)
    print(gs.best_params_)


# --------------------------------------------------------
# K Nearest Neighbours
def knn_classification(X_train, y_train, X_test, y_test):
    # check_knn_parameters(X_train, y_train)
    knn = KNeighborsClassifier(n_neighbors=13, metric='manhattan')
    knn.fit(X_train, y_train)
    print(f"(KNN) Train accuracy {knn.score(X_train, y_train)}\nTest accuracy {knn.score(X_test, y_test)}")


def check_knn_parameters(X, y):
    neighbours = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    metrics = ['minkowski', 'manhattan']
    param_grid = {'n_neighbors': neighbours,
                  'metric': metrics}
    gs = GridSearchCV(estimator=KNeighborsClassifier(),
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=10,
                      n_jobs=-1)
    gs.fit(X, y)
    print(gs.best_score_)
    print(gs.best_params_)


# --------------------------------------------------------
# SVM
def svm_classification(X_train, y_train, X_test, y_test):
    # check_svm_parameters(X_train, y_train)
    svm = SVC(C=0.5, kernel='rbf', random_state=1)
    svm.fit(X_train, y_train)
    print(f"(KNN) Train accuracy {svm.score(X_train, y_train)}\nTest accuracy {svm.score(X_test, y_test)}")


def check_svm_parameters(X, y):
    C = [0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0]
    kernels = ['rbf', 'linear', 'poly', 'sigmoid']
    param_grid = {'C': C,
                  'kernel': kernels}
    gs = GridSearchCV(estimator=SVC(random_state=1),
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=10,
                      n_jobs=-1)
    gs.fit(X, y)
    print(gs.best_score_)
    print(gs.best_params_)


# --------------------------------------------------------
# RANDOM FOREST
def log_reg_classification(X_train, y_train, X_test, y_test):
    # check_log_reg_parameters(X_train, y_train)
    lr = LogisticRegression(penalty='none', solver='lbfgs', multi_class='ovr', random_state=1)
    lr.fit(X_train, y_train)
    print(f"(LOG. REG.) Train accuracy {lr.score(X_train, y_train)}\nTest accuracy {lr.score(X_test, y_test)}")


def check_log_reg_parameters(X, y):
    penalties = ['l2', 'none']
    solvers = ['lbfgs']
    param_grid = {'penalty': penalties,
                  'solver': solvers}
    gs = GridSearchCV(estimator=LogisticRegression(random_state=1),
                      param_grid=param_grid,
                      scoring='accuracy',
                      cv=10,
                      n_jobs=-1)
    gs.fit(X, y)
    print(gs.best_score_)
    print(gs.best_params_)
