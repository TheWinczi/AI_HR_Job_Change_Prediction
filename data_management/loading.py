import pandas as pd
from .processing import *
from .utilities import *


def load_native_data(path: str):
    """
    Load native .csv data from path.

    Parameters
    ----------
    path : str
        File path.

    Raises
    ------
    IOError
        If file doesn't exists or access to file was denied.

    Returns
    -------
    data
        loaded data from file or None is something goes wrong.
    """
    native_data = None
    try:
        native_data = pd.read_csv(path)
        native_data = fill_nans_in_all_columns(native_data)
    except IOError:
        raise(IOError("Loading file failed"))

    return native_data


def load_processed_data(path: str):
    """
    Load data from file and process they - cast all values into numbers.

    Parameters
    ----------
    path : str
        File path.

    Returns
    -------
    data
        Processed data.
    """
    native_data = load_native_data(path)
    processed_data = process_native_data(native_data)
    return processed_data


def load_train_test_data(path: str):
    """
    Load data from file, process them and split into train/test sets ready to train and test.

    Parameters
    ----------
    path : str
        File path.

    Returns
    -------
    (x_train, y_train, x_test, y_test)
        Tuple of train/test data ready to train and test.
    """
    processed_data = load_processed_data(path)
    return prepare_train_test_data(processed_data)
