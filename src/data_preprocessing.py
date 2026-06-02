import os
import sys
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config.paths_config import *
from utils.common_functions import read_csv, save_csv, read_yaml

from src.logger import logging
from src.exception import CustomException

class DataPreprocessing:
    def __init__(self, config, input_file_path, output_file_path):
        self.raw_file_path = input_file_path
        self.preprocessed_file_path = output_file_path

        self.config = read_yaml(config)
        self.test_size = self.config['data_preprocessing']['test_size']
        self.random_state = self.config['data_preprocessing']['random_state']
        self.target_column = self.config['data_preprocessing']['target_column']

        os.makedirs(self.preprocessed_file_path, exist_ok=True)

    def load_data(self):
        try:
            logging.info(f"Loading data from {self.raw_file_path}")

            df = read_csv(self.raw_file_path)

            logging.info(f"Data loaded successfully from {self.raw_file_path}")
            return df
        except Exception as e:
            logging.error("Error while loading the data")
            raise CustomException(e, sys)

    def preprocess_data(self, df):
        try:
            logging.info("Transforming Timestamp to datetime format")

            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

            logging.info("Extracting date features from Timestamp")

            df['Year'] = df['Timestamp'].dt.year.astype(int)
            df['Month'] = df['Timestamp'].dt.month.astype(int)
            df['Day'] = df['Timestamp'].dt.day.astype(int)

            df['Hour'] = df['Timestamp'].dt.hour.astype(int)
            df['Minute'] = df['Timestamp'].dt.minute.astype(int)
            df['Second'] = df['Timestamp'].dt.second.astype(int)

            logging.info("Dropping the original Timestamp and MachineID columns")

            df.drop('Timestamp', axis=1, inplace=True)
            df.drop('Machine_ID', axis=1, inplace=True)

            logging.info("Preprocessing completed successfully")
            return df

        except Exception as e:
            logging.error("Error while preprocessing the data")
            raise CustomException(e, sys)

    def encode_data(self, df):
        try:
            logging.info("Encoding the categorical features")

            cat_features = df.select_dtypes(include=['str', 'object']).columns
            label_encoders = {}

            for col in cat_features:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le

            logging.info("Encoding completed successfully")

            return df, label_encoders

        except Exception as e:
            logging.error("Error while encoding the data")
            raise CustomException(e, sys)

    def split_data(self, df):
        try:
            logging.info("Splitting the data into train and test sets")

            X = df.drop(self.target_column, axis=1)
            y = df[self.target_column]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

            logging.info("Data splitting completed successfully")
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error("Error while splitting the data")
            raise CustomException(e, sys)

    def scale_data(self, X_train, X_test):
        try:
            logging.info("Scaling the features using StandardScaler")

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            logging.info("Feature scaling completed successfully")
            return X_train, X_test, scaler

        except Exception as e:
            logging.error("Error while scaling the data")
            raise CustomException(e, sys)

    def run(self):
        try:
            logging.info("Starting data preprocessing")

            df = self.load_data()
            df = self.preprocess_data(df)
            df, label_encoders = self.encode_data(df)

            X_train, X_test, y_train, y_test = self.split_data(df)

            X_train, X_test, scaler = self.scale_data(X_train, X_test)

            save_csv(X_train, X_TRAIN_PATH)
            save_csv(X_test, X_TEST_PATH)
            save_csv(y_train, Y_TRAIN_PATH)
            save_csv(y_test, Y_TEST_PATH)

            joblib.dump(label_encoders, LABEL_ENCODER_PATH)
            joblib.dump(scaler, SCALER_PATH)

            logging.info("Data preprocessing completed successfully")
            logging.info(f"Preprocessed data saved successfully at {self.preprocessed_file_path}")

        except Exception as e:
            logging.error("Error while running the data preprocessing")
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_preprocessing = DataPreprocessing(CONFIG_PATH, RAW_FILE_PATH, PREPROCESSED_DIR)
    data_preprocessing.run()
