import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def process_native_data(df: pd.DataFrame):
    processed_df = process_all_labels(df.copy())
    return processed_df


def process_all_labels(df: pd.DataFrame):
    dummies_cols = ["gender", "major_discipline", "training_hours",
                    "last_new_job", "company_type", "company_size", "experience"]
    insignificant_cols = ["enrollee_id", "city"]

    df = process_insignificant_columns(df, dummies_cols)
    df.drop(insignificant_cols + dummies_cols, axis=1)

    df["relevent_experience"] = process_relevent_exp(df)
    df["enrolled_university"] = process_enrolled_university(df)
    df["education_level"] = process_education_level(df)

    return df


def process_insignificant_columns(df: pd.DataFrame, columns: list[str]):
    processed_cols = pd.get_dummies(df[columns], drop_first=True)
    for col in processed_cols.columns:
        df[col] = processed_cols[col]
    return df


def process_relevent_exp(df: pd.DataFrame):
    """ There is light correlation between gender and target """
    exp_mapping = {"No relevent experience": 0,
                   "Has relevent experience": 1}
    exp = df["relevent_experience"]
    df["relevent_experience"] = exp.map(exp_mapping)
    return df["relevent_experience"]


def process_enrolled_university(df: pd.DataFrame):
    """ There is light correlation between gender and target """
    university_mapping = {np.nan: 0,
                          "no_enrollment": 1,
                          "Part time course": 2,
                          "Full time course": 3}
    values = df["enrolled_university"]
    df["enrolled_university"] = values.map(university_mapping)
    return df["enrolled_university"]


def process_education_level(df: pd.DataFrame):
    """ There is light correlation between gender and target """
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
    no_target_cols = list(filter(lambda x: x != "target", df.columns))
    X = df[no_target_cols].values
    y = df["target"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    return X_train_std, y_train, X_test_std, y_test
