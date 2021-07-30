import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from plotting.histograms import *


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

    def analyse_native_data(self):
        encoded_df = self.cast_all_columns_values_into_uniques()
        no_target_labels = list(filter(lambda item: item != "target", self.native_data.columns))
        no_target_values = encoded_df[no_target_labels].values
        no_target_values = StandardScaler().fit_transform(no_target_values)
        plot_LDA_features_importances(no_target_values,
                                      encoded_df["target"].values,
                                      titles=["PCA", "KernelPCA", "LDA"],
                                      xlabels=["Index of main component" for _ in range(3)],
                                      ylabels=["explained variance coef" for _ in range(3)],
                                      suptitle="Dimensionality reductions, features importances")

        cols_labels = np.array(self.native_data.columns)
        cols_labels = cols_labels[cols_labels != "enrollee_id"]

        fig_cols = 3
        fig_rows = np.ceil(len(cols_labels) / fig_cols).astype(np.int32)

        plt.figure(figsize=(16, 9))
        for i, label in enumerate(cols_labels):
            if label not in ["target"]:
                plt.subplot(fig_rows - 1, fig_cols, i + 1)
                x = self._fill_nans(self.native_data[label]).values
                y = self.native_data["target"].values
                draw_correlation_histogram(x, y, title=label)
        plt.suptitle("Correlations histograms")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(16, 9))
        for i, label in enumerate(cols_labels):
            plt.subplot(fig_rows, fig_cols, i + 1)
            histogram_data = self._fill_nans(self.native_data[label]).values
            draw_counts_histogram(histogram_data, title=label)
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

    def cast_all_columns_values_into_uniques(self):
        df = self.native_data.copy()
        le = LabelEncoder()

        for column in df.columns:
            if df[column].dtype not in [np.int32, np.int64, np.float]:
                df[column] = le.fit_transform(df[column])

        return df
