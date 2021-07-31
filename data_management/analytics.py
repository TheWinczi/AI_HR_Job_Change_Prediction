import pandas as pd
from .utilities import *
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from plotting import *


def analyse_native_data(df: pd.DataFrame):
    encoded_df = cast_all_columns_values_into_unique_labels(df)
    no_target_labels = list(filter(lambda item: item != "target", df.columns))
    no_target_values = encoded_df[no_target_labels].values
    no_target_values = StandardScaler().fit_transform(no_target_values)
    plot_PCA_features_importances(no_target_values,
                                  title="PCA algorithm results",
                                  xlabel="Index of main component",
                                  ylabel="explained variance coef")

    cols_labels = np.array(df.columns)
    cols_labels = cols_labels[cols_labels != "enrollee_id"]

    fig_cols = 3
    fig_rows = np.ceil(len(cols_labels) / fig_cols).astype(np.int32)

    plt.figure(figsize=(16, 9))
    for i, label in enumerate(cols_labels):
        if label not in ["target"]:
            plt.subplot(fig_rows - 1, fig_cols, i + 1)
            x = fill_nans_in_column(df[label]).values
            y = df["target"].values
            draw_correlation_histogram(x, y, title=label)
    plt.suptitle("Correlations histograms")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(16, 9))
    for i, label in enumerate(cols_labels):
        plt.subplot(fig_rows, fig_cols, i + 1)
        histogram_data = fill_nans_in_column(df[label]).values
        draw_counts_histogram(histogram_data, title=label)
    plt.suptitle("Counts histograms")
    plt.tight_layout()
    plt.show()
