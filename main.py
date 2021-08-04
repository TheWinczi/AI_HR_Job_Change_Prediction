import os
from classification import *
from data_management import *


def main():
    native_data_file_path = os.path.join("data", "train.csv")

    native_data = load_native_data(native_data_file_path)
    processed_data = process_native_data(native_data)
    X_train, y_train, X_test, y_test = prepare_train_test_data(processed_data)

    # analyse_native_data(native_data)

    try_all_classifiers(X_train, y_train, X_test, y_test)

    # tree_classification(X_train, y_train, X_test, y_test)
    # random_forest_classification(X_train, y_train, X_test, y_test)
    # knn_classification(X_train, y_train, X_test, y_test)
    # svm_classification(X_train, y_train, X_test, y_test)
    # log_reg_classification(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    main()
