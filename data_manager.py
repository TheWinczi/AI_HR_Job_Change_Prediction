import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import KernelPCA


class DataManager:

    processed_data_file = "processed_data.csv"

    def __init__(self):
        self.native_data = None
        self.processed_data = None

    def load_data(self):
        self.native_data = pd.read_csv("data/train.csv")

    def load_processed_data(self):
        self.load_data()
        path = os.path.join('data', self.processed_data_file)
        if os.path.exists(path):
            self.processed_data = pd.read_csv(path)
        else:
            self.process_native_data()
            self.save_processed_data()
        return self.processed_data

    def process_native_data(self):
        self.processed_data = self.native_data.copy()
        self.process_all_labels()
        self.processed_data = self.processed_data.drop(["enrollee_id"], axis=1)

    def process_all_labels(self):
        ins_cols = ["city", "gender", "major_discipline", "training_hours",
                    "last_new_job", "company_type", "company_size", "experience"]
        self.process_insignificant_data(ins_cols)
        self.drop_insignificant_columns(ins_cols + ["city"])
        self.process_relevent_exp()
        self.process_enrolled_university()
        self.process_education_level()

    def process_insignificant_data(self, columns: list[str]):
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

    def drop_insignificant_columns(self, columns: list[str]):
        self.processed_data = self.processed_data.drop(columns, axis=1)

    def save_processed_data(self):
        path = os.path.join('data', self.processed_data_file)
        try:
            self.processed_data.to_csv(path, index=False)
        except IOError:
            print('Saving file failed')

    def prepare_train_test_data(self):
        cols = list(filter(lambda x: x != "target", self.processed_data.columns))
        X = self.processed_data[cols].values
        y = self.processed_data["target"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

        sc = StandardScaler()
        sc.fit(X_train)
        X_train_std = sc.transform(X_train)
        X_test_std = sc.transform(X_test)

        kpca = KernelPCA(n_components=10, kernel='rbf', gamma=15)
        X_train_pca = kpca.fit_transform(X_train_std)
        X_test_pca = kpca.fit_transform(X_test_std)
        print(len(X_train_pca[0]))

        return X_train_pca, y_train, X_test_pca, y_test

    @staticmethod
    def show_correlation(x_data: list, x_ticks: list[str], target: list, title: str, xlabel: str, ylabel: str):
        sums = []
        ticks = np.unique(x_data)
        for tic in ticks:
            indices = np.where(x_data == tic)
            sums.append(target[indices].sum() / indices[0].size)

        fig = plt.figure()
        plt.bar(x_ticks, sums)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()


def process_city_id(self):
    """ There is not correlation between gender and target """
    le = LabelEncoder()
    values = self.processed_train_data["city"].values
    le.fit(values)
    self.processed_train_data["city"] = le.transform(values)


def process_gender(self):
    """ There is not correlation between gender and target """
    le = LabelEncoder()
    values = self.processed_train_data["gender"].values
    le.fit(values)
    self.processed_train_data["gender"] = le.transform(values)


def process_major_discipline(self):
    """ There is not correlation between gender and target """
    le = LabelEncoder()
    values = self.processed_train_data["major_discipline"].values
    le.fit(values)
    self.processed_train_data["major_discipline"] = le.transform(values)


def process_experience(self):
    """ There is not correlation between gender and target """
    le = LabelEncoder()
    values = self.processed_train_data["experience"].values
    le.fit(values)
    self.processed_train_data["experience"] = le.transform(values)


def process_company_size(self):
    """ There is NOT correlation between gender and target """
    le = LabelEncoder()
    values = self.processed_train_data["company_size"].values
    le.fit(values)
    self.processed_train_data["company_size"] = le.transform(values)


def process_company_type(self):
    """ There is NOT correlation between company type and target """
    le = LabelEncoder()
    values = self.processed_train_data["company_type"].values
    le.fit(values)
    self.processed_train_data["company_type"] = le.transform(values)


def process_last_new_job(self):
    """ There is NOT correlation between last new job and target """
    le = LabelEncoder()
    values = self.processed_train_data["last_new_job"].values
    le.fit(values)
    self.processed_train_data["last_new_job"] = le.transform(values)


def process_training_hours(self):
    """ There is NOT correlation between training hours and target """
    le = LabelEncoder()
    values = self.processed_train_data["training_hours"].values
    le.fit(values)
    self.processed_train_data["training_hours"] = le.transform(values)
