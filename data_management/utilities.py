import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def fill_nans_in_all_columns(df: pd.DataFrame):
    columns = df.columns
    for column in columns:
        df[column] = fill_nans_in_column(df[column])
    return df


def fill_nans_in_column(df: pd.DataFrame):
    if df.values.dtype == np.float:
        df = df.fillna(0.0)
    elif df.values.dtype == np.int32:
        df = df.fillna(0)
    else:
        df = df.fillna("None")
    return df


def cast_all_columns_values_into_unique_labels(df: pd.DataFrame):
    for column in df.columns:
        df[column] = cast_column_values_into_unique_labels(df[column])
    return df


def cast_column_values_into_unique_labels(df: pd.DataFrame):
    le = LabelEncoder()

    if df.dtype not in [np.int32, np.int64, np.float]:
        df = le.fit_transform(df)
    return df
