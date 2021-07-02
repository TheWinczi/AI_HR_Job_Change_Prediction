import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


class DataManager:

    def __init__(self):
        self.native_train_data = None
        self.native_test_data = None
        self.processed_train_data = None
        self.processed_test_data = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        self.native_train_data = pd.read_csv("data/train.csv")
        self.native_test_data = pd.read_csv("data/test.csv")

    def load_processed_data(self):
        self.load_data()
        self.processed_train_data = self.native_train_data.copy()
        self.processed_test_data = self.native_test_data.copy()
        self.process_all_labels()
        return self.processed_train_data, self.processed_test_data

    def process_all_labels(self):
        ins = ["city", "gender", "major_discipline", "training_hours",
               "last_new_job", "company_type", "company_size", "experience"]
        self.process_insignificant_data(ins)
        self.process_relevent_exp()
        self.process_enrolled_university()
        self.process_education_level()

    def process_insignificant_data(self, columns: list[str]):
        processed_cols = pd.get_dummies(self.processed_train_data[columns], drop_first=True)
        for col in processed_cols.columns:
            self.processed_train_data[col] = processed_cols[col]

    def process_relevent_exp(self):
        """ There is light correlation between gender and target """
        exp_mapping = {"No relevent experience": 0,
                       "Has relevent experience": 1}
        exp = self.processed_train_data["relevent_experience"]
        self.processed_train_data["relevent_experience"] = exp.map(exp_mapping)

    def process_enrolled_university(self):
        """ There is light correlation between gender and target """
        university_mapping = {np.nan: 0,
                              "no_enrollment": 1,
                              "Part time course": 2,
                              "Full time course": 3}
        values = self.processed_train_data["enrolled_university"]
        self.processed_train_data["enrolled_university"] = values.map(university_mapping)

    def process_education_level(self):
        """ There is light correlation between gender and target """
        education_mapping = {np.nan: 0,
                             "High School": 1,
                             "Masters": 2,
                             "Graduate": 3}
        values = self.processed_train_data["education_level"]
        self.processed_train_data["education_level"] = values.map(education_mapping)

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
