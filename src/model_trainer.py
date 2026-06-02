import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix

from utils.common_functions import read_csv, read_yaml
from config.paths_config import *

from src.logger import logging
from src.exception import CustomException

class ModelTrainer:
    def __init__(self, config, output_path):
        self.config = read_yaml(config)
        self.penalty = self.config['model_training']['penalty']
        self.C = self.config['model_training']['C']
        self.solver = self.config['model_training']['solver']
        self.random_state = self.config['model_training']['random_state']

        self.output_path = output_path

        os.makedirs(self.output_path, exist_ok=True)

    def load_data(self):
        try:
            logging.info("Loading preprocessed data")

            X_train = read_csv(X_TRAIN_PATH).values
            y_train = read_csv(Y_TRAIN_PATH).values.ravel()
            X_test = read_csv(X_TEST_PATH).values
            y_test = read_csv(Y_TEST_PATH).values.ravel()

            logging.info("Successfully loaded preprocessed data")
            return X_train, y_train, X_test, y_test

        except Exception as e:
            logging.error("Error while loading preprocessed data")
            raise CustomException(e, sys)

    def train_model(self, X_train, y_train):
        try:
            logging.info("Training the model")

            model = LogisticRegression(penalty=self.penalty, C=self.C, solver=self.solver, random_state=self.random_state)
            model.fit(X_train, y_train)

            logging.info("Model training completed successfully")
            return model

        except Exception as e:
            logging.error("Error while training the model")
            raise CustomException(e, sys)

    def evaluate_model(self, model, X_test, y_test):
        try:
            logging.info("Evaluating the model")

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            classification_rep = classification_report(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)

            logging.info("Model evaluation completed successfully")
            return accuracy, recall, precision, f1, classification_rep, conf_matrix

        except Exception as e:
            logging.error("Error while evaluating the model")
            raise CustomException(e, sys)

    def save(self):
        try:
            logging.info("Saving the model and evaluation metrics")

            X_train, y_train, X_test, y_test = self.load_data()
            model = self.train_model(X_train, y_train)
            accuracy, recall, precision, f1, classification_rep, conf_matrix = self.evaluate_model(model, X_test, y_test)

            os.makedirs(self.output_path, exist_ok=True)
            joblib.dump(model, MODEL_PATH)

            with open(MODEL_METRICS_PATH, 'w') as f:
                f.write(f"Accuracy: {accuracy}\n")
                f.write(f"Recall: {recall}\n")
                f.write(f"Precision: {precision}\n")
                f.write(f"F1 Score: {f1}\n")
                f.write("Classification Report:\n")
                f.write(classification_rep)

            logging.info(f"Model saved successfully at {MODEL_PATH}")
            logging.info(f"Evaluation metrics saved successfully at {self.output_path}")
            logging.info("Creating confusion matrix heatmap")

            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix Heatmap')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.savefig(CONF_MATRIX_PATH)
            plt.close()

            logging.info(f"Confusion matrix heatmap saved successfully at {CONF_MATRIX_PATH}")

        except Exception as e:
            logging.error("Error while saving the model and evaluation metrics")
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        model_trainer = ModelTrainer(config=CONFIG_PATH, output_path=MODEL_DIR)
        model_trainer.save()

    except Exception as e:
        logging.error("Error in model training process")
        raise CustomException(e, sys)
