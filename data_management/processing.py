import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from .utilities import drop_columns


def process_native_data(df: pd.DataFrame):
    """
    Process native data. Process all columns in DataFrame to
    allow machine learning.

    Parameters
    ----------
    df : DataFrame
        DataFrame of all needed data.

    Returns
    -------
    proc_data
        Processed DataFrame with data ready to machine learning.
    """
    processed_df = process_all_labels(df.copy())
    return processed_df


def process_all_labels(df: pd.DataFrame):
    """
    Process all labels (columns values) in DataFrame to allow
    machine learning

    Parameters
    ----------
    df : DataFrame
        DataFrame of all needed data.

    Returns
    -------
    proc_data
        Processed DataFrame with data ready to machine learning.
    """
    dummies_cols = ["gender", "major_discipline", "training_hours",
                    "last_new_job", "company_type", "company_size", "experience"]
    encoded_cols = ["city"]
    insignificant_cols = ["enrollee_id", "city"]

    df = process_dummies_columns(df, dummies_cols)
    df = drop_columns(df, insignificant_cols)

    df["relevent_experience"] = process_relevent_exp(df)
    df["enrolled_university"] = process_enrolled_university(df)
    df["education_level"] = process_education_level(df)

    return df


def process_dummies_columns(df: pd.DataFrame, columns: list[str]):
    """
    Process DataFrame columns using dummies/one_hot_encoding.

    Parameters
    ----------
    df : DataFrame
        DataFrame with columns to process.

    columns : list[str]
        List of columns labels to process.

    Returns
    -------
    df
        DataFrame with processed columns.
    """
    processed_cols = pd.get_dummies(df[columns], drop_first=True)
    for col in processed_cols.columns:
        df[col] = processed_cols[col]
    df = df.drop(columns, axis=1)
    return df


def encode_labels(df: pd.DataFrame, columns: list[str]):
    """
    Using LabelEncoder encode all values in columns into numeric unique values.

    Parameters
    ----------
    df : DataFrame
        DataFrame with columns to encode.

    columns : list[str]
        List of columns labels to encode.

    Returns
    -------
    df
        DataFrame with encoded columns values.
    """
    le = LabelEncoder()
    for column in columns:
        df[column] = le.fit_transform(df[column].values)
    return df


def process_relevent_exp(df: pd.DataFrame):
    """
    Process DataFrame "relevent_experience" column using created mapping dict.

    Parameters
    ----------
    df : DataFrame
        DataFrame with column to process.

    Returns
    -------
    df
        Processed "relevent_experience" column as DataFrame.
    """
    exp_mapping = {"No relevent experience": 0,
                   "Has relevent experience": 1}
    exp = df["relevent_experience"]
    df["relevent_experience"] = exp.map(exp_mapping)
    return df["relevent_experience"]


def process_enrolled_university(df: pd.DataFrame):
    """
    Process DataFrame "enrolled_university" column using created mapping dict.

    Parameters
    ----------
    df : DataFrame
        DataFrame with column to process.

    Returns
    -------
    df
        Processed "enrolled_university" column as DataFrame.
    """
    university_mapping = {np.nan: 0,
                          "no_enrollment": 1,
                          "Part time course": 2,
                          "Full time course": 3}
    values = df["enrolled_university"]
    df["enrolled_university"] = values.map(university_mapping)
    return df["enrolled_university"]


def process_education_level(df: pd.DataFrame):
    """
    Process DataFrame "education_level" column using created mapping dict.

    Parameters
    ----------
    df : DataFrame
        DataFrame with column to process.

    Returns
    -------
    df
        Processed "education_level" column as DataFrame.
    """
    education_mapping = {np.nan: 0,
                         "Primary School": 1,
                         "High School": 2,
                         "Masters": 3,
                         "Graduate": 4,
                         "Phd": 5}
    values = df["education_level"]
    df["education_level"] = values.map(education_mapping)
    return df["education_level"]


def prepare_train_test_data(df: pd.DataFrame):
    """
    Prepare tran and test sets ready to machine learning.

    Parameters
    ----------
    df : DataFrame
        DataFrame with train/test data.

    Returns
    -------
    (X_train, y_train, X_test, y_test)
        Tuple of split data.
    """
    no_target_cols = list(filter(lambda x: x != "target", df.columns))

    X = df[no_target_cols].values
    y = df["target"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    return X_train_std, y_train, X_test_std, y_test
