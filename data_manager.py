import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mlxtend.plotting import scatterplotmatrix


class DataManager(object):
    PROCESSED_DATA_FILE_PATH = os.path.join("data", "processed_data.csv")
    NATIVE_DATA_FILE_PATH = os.path.join("data", "train.csv")

    def __init__(self):
        self.native_data = None
        self.processed_data = None

    def load_data(self):
        self.native_data = pd.read_csv(self.NATIVE_DATA_FILE_PATH)
        return self.native_data

    def load_processed_data(self):
        self.load_data()
        if os.path.exists(self.PROCESSED_DATA_FILE_PATH):
            self.processed_data = pd.read_csv(self.PROCESSED_DATA_FILE_PATH)
        else:
            self.process_native_data()
            self.save_processed_data()
        return self.processed_data

    def load_train_test_data(self):
        if self.processed_data is None:
            self.load_processed_data()
        return self.prepare_train_test_data()

    def plot_native_data_scatter_matrix(self):
        cols = self.native_data.columns
        scatterplotmatrix(self.native_data[cols].values,
                          figsize=(16, 9),
                          names=cols,
                          alpha=0.5)

    def process_native_data(self):
        self.processed_data = self.native_data.copy()
        self.process_all_labels()

    def process_all_labels(self):
        dummies_cols = ["gender", "major_discipline", "training_hours",
                        "last_new_job", "company_type", "company_size", "experience"]
        insignificant_cols = ["enrollee_id", "city"]

        self.process_insignificant_columns(dummies_cols)
        self.processed_data.drop(insignificant_cols + dummies_cols, axis=1)

        self.process_relevent_exp()
        self.process_enrolled_university()
        self.process_education_level()

    def process_insignificant_columns(self, columns: list[str]):
        processed_cols = pd.get_dummies(self.processed_data[columns], drop_first=True)
        for col in processed_cols.columns:
            self.processed_data[col] = processed_cols[col]

    def process_relevent_exp(self):
        """ There is light correlation between gender and target """
        exp_mapping = {"No relevent experience": 0,
                       "Has relevent experience": 1}
        exp = self.processed_data["relevent_experience"]
        self.processed_data["relevent_experience"] = exp.map(exp_mapping)

    def process_enrolled_university(self):
        """ There is light correlation between gender and target """
        university_mapping = {np.nan: 0,
                              "no_enrollment": 1,
                              "Part time course": 2,
                              "Full time course": 3}
        values = self.processed_data["enrolled_university"]
        self.processed_data["enrolled_university"] = values.map(university_mapping)

    def process_education_level(self):
        """ There is light correlation between gender and target """
        education_mapping = {np.nan: 0,
                             "Primary School": 1,
                             "High School": 2,
                             "Masters": 3,
                             "Graduate": 4,
                             "Phd": 5}
        values = self.processed_data["education_level"]
        self.processed_data["education_level"] = values.map(education_mapping)


    def save_processed_data(self):
        try:
            self.processed_data.to_csv(self.PROCESSED_DATA_FILE_PATH, index=False)
        except IOError:
            print('Saving file with processed data failed')


    def prepare_train_test_data(self):
        cols = list(filter(lambda x: x != "target", self.processed_data.columns))
        X = self.processed_data[cols].values
        y = self.processed_data["target"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1, stratify=y)

        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)

        return X_train_std, y_train, X_test_std, y_test


    @staticmethod
    def draw_counts_histogram(data: np.ndarray,
                              title: str = None,
                              xlabel: str = None,
                              ylabel: str = None):
        sums = []
        unique_ticks = np.unique(data)
        for tic in unique_ticks:
            sums.append((data == tic).sum())

        length = len(sums)
        colors = list(zip(np.random.rand(length), np.random.rand(length), np.random.rand(length), [1 for _ in sums]))
        plt.bar(np.arange(length), sums, color=colors)
        plt.xticks(np.arange(length), unique_ticks)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()


    @staticmethod
    def draw_correlation_histogram(x: np.ndarray,
                                   y: np.ndarray,
                                   title: str = None,
                                   xlabel: str = None,
                                   ylabel: str = None):
        targets = np.unique(y)
        ticks = np.unique(x)

        bars_values = []
        targets_counts = [0 for _ in ticks]

        for target in targets:
            sums = []
            for i, tick in enumerate(ticks):
                indices = np.logical_and(x == tick, y == target)
                indices_sum = sum(indices)
                sums.append(indices_sum)
                targets_counts[i] += indices_sum
            bars_values.append(sums.copy())

        bottom_values = [0 for _ in range(len(ticks))]
        for i in range(len(targets)):
            for j, count in enumerate(targets_counts):
                bars_values[i][j] = bars_values[i][j] / count
            plt.bar(np.arange(len(ticks)), bars_values[i], bottom=bottom_values)
            bottom_values = bars_values[i]

        plt.xticks(np.arange(len(ticks)), ticks)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()


    def analyse_native_data(self):
        cols_labels = np.array(self.native_data.columns)
        cols_labels = cols_labels[cols_labels != "enrollee_id"]

        fig_cols = 3
        fig_rows = np.ceil(len(cols_labels) / fig_cols).astype(np.int32)

        plt.figure(figsize=(16, 9))
        for i, label in enumerate(cols_labels):
            if label not in ["target"]:
                plt.subplot(fig_rows-1, fig_cols, i + 1)
                x = self._fill_nans(self.native_data[label]).values
                y = self.native_data["target"].values
                self.draw_correlation_histogram(x, y, title=label)
        plt.suptitle("Correlations histograms")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(16, 9))
        for i, label in enumerate(cols_labels):
            plt.subplot(fig_rows, fig_cols, i+1)
            histogram_data = self._fill_nans(self.native_data[label]).values
            self.draw_counts_histogram(histogram_data, title=label)
        plt.suptitle("Counts histograms")
        plt.tight_layout()
        plt.show()


    def _fill_nans(self, df: pd.DataFrame):
        if df.values.dtype == np.float:
            df = df.fillna(0.0)
        elif df.values.dtype == np.int32:
            df = df.fillna(0)
        else:
            df = df.fillna("None")
        return df
