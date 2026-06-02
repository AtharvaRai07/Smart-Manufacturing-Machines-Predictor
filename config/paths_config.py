import os

################################# DATA INGESTION #################################
RAW_DIR = "artifacts/raw"
RAW_FILE_PATH = os.path.join(RAW_DIR, "data.csv")

CONFIG_PATH = "config/config.yaml"

################################ DATA PREPROCESSING #################################
PREPROCESSED_DIR = "artifacts/preprocessed"
X_TRAIN_PATH = os.path.join(PREPROCESSED_DIR, "X_train.csv")
Y_TRAIN_PATH = os.path.join(PREPROCESSED_DIR, "y_train.csv")
X_TEST_PATH = os.path.join(PREPROCESSED_DIR, "X_test.csv")
Y_TEST_PATH = os.path.join(PREPROCESSED_DIR, "y_test.csv")

LABEL_ENCODER_PATH = os.path.join(PREPROCESSED_DIR, "label_encoder.pkl")
SCALER_PATH = os.path.join(PREPROCESSED_DIR, "scaler.pkl")

################################# MODEL TRAINING #################################
MODEL_DIR = "artifacts/model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
MODEL_METRICS_PATH = os.path.join(MODEL_DIR, "metrics.txt")
CONF_MATRIX_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")
