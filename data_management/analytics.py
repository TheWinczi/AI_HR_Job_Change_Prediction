import pandas as pd
from sklearn.preprocessing import StandardScaler
from plotting import *
from .utilities import *


def analyse_native_data(df: pd.DataFrame):
    """
    Perform all implemented input data analysis.
    Several charts will be created to display
    the results of the analysis.

    Parameters
    ----------
    df : DataFrame
        DataFrame object storing all data to analyze.
    """
    analyse_data_columns_count(df, sort=False)
    analyse_data_target_dependencies(df, sort=True)
    analyse_data_PCA_reduction(df)


def analyse_data_target_dependencies(df: pd.DataFrame,
                                     sort: bool = False):
    """
    Analyze data dependencies on target. Show is any dependencies
    between specific columns values and target.

    Parameters
    ----------
    df : DataFrame
        DataFrame object storing all data to analyze.

    sort : bool {default: False}
        Are result has to be sorted.
    """
    no_target_labels = list(filter(lambda item: item not in ["enrollee_id", "target"], df.columns.tolist()))

    target_df = df["target"]
    no_target_df = df[no_target_labels]
    x, y = no_target_df.values, target_df.values

    plot_data_target_dependencies(x, y,
                                  titles=no_target_df.columns.tolist(),
                                  suptitle=f"Dependencies of individual data columns on the target (sorted={sort})",
                                  sort=sort)


def analyse_data_columns_count(df: pd.DataFrame,
                               sort: bool = False):
    """
    Analyze data columns values counts.

    Parameters
    ----------
    df : DataFrame
        DataFrame object storing all data to analyze.

    sort : bool {default: False}
        Are result has to be sorted.
    """
    no_target_labels = list(filter(lambda item: item not in ["enrollee_id"], df.columns.tolist()))

    columns_df = df[no_target_labels]
    x = columns_df.values

    plot_data_columns_counts(x,
                             titles=columns_df.columns.tolist(),
                             ylabels=["count" for _ in range(len(no_target_labels))],
                             suptitle=f"Columns values counts (sorted={sort})",
                             sort=sort)


def analyse_data_PCA_reduction(df: pd.DataFrame):
    """
    Analyze data features importances using PCA dimensionality
    reduction algorithm. By design, sorted in descending order.

    Parameters
    ----------
    df : DataFrame
        DataFrame object storing all data to analyze.
    """
    encoded_df = cast_all_columns_values_into_unique_labels(df)

    no_target_labels = list(filter(lambda item: item != "target", df.columns))
    no_target_values = encoded_df[no_target_labels].values
    no_target_values = StandardScaler().fit_transform(no_target_values)

    plot_PCA_features_importances(no_target_values,
                                  title="PCA algorithm results",
                                  xlabel="Index of main component",
                                  ylabel="explained variance coef")
