from data_manager import *
from classifiers import *



def main():
    dm = DataManager()
    data = dm.load_processed_data()
    X_train, y_train, X_test, y_test = dm.prepare_train_test_data()

    tree_classification(X_train, y_train, X_test, y_test)
    print()
    random_forest_classification(X_train, y_train, X_test, y_test)
    print()
    knn_classification(X_train, y_train, X_test, y_test)
    print()
    svm_classification(X_train, y_train, X_test, y_test)
    print()
    log_reg_classification(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()
