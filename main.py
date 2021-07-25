from data_manager import *
from classifiers import *



def main():
    dm = DataManager()

    dm.load_data()
    dm.analyse_native_data()

    X_train, y_train, X_test, y_test = dm.load_train_test_data()

    tree_classification(X_train, y_train, X_test, y_test)
    random_forest_classification(X_train, y_train, X_test, y_test)
    knn_classification(X_train, y_train, X_test, y_test)
    svm_classification(X_train, y_train, X_test, y_test)
    log_reg_classification(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()
