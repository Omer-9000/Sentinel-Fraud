"""
config.py — Central Configuration for SentinelFraud Detection System
Centralizes all paths, hyperparameters, and thresholds so nothing is hardcoded.
"""

import os
from pathlib import Path

# Project Root
BASE_DIR = Path(__file__).resolve().parent

# Data Paths
DATA_DIR          = BASE_DIR / "data"
RAW_DATA_DIR      = DATA_DIR / "raw"
PROCESSED_DATA_DIR= DATA_DIR / "processed"

RAW_DATA_PATH     = RAW_DATA_DIR / "creditcard.csv"
PROCESSED_PATH    = PROCESSED_DATA_DIR / "transactions_processed.parquet"
TRAIN_PATH        = PROCESSED_DATA_DIR / "train.parquet"
TEST_PATH         = PROCESSED_DATA_DIR / "test.parquet"

# Model Paths
MODEL_DIR         = BASE_DIR / "models" / "saved"
XGBOOST_PATH      = MODEL_DIR / "xgboost_model.joblib"
ISOLATION_FOREST_PATH = MODEL_DIR / "isolation_forest.joblib"
SCALER_PATH       = MODEL_DIR / "scaler.joblib"
FEATURE_LIST_PATH = MODEL_DIR / "feature_columns.joblib"

# Reports / Logs 
REPORTS_DIR       = BASE_DIR / "reports"
LOGS_DIR          = BASE_DIR / "logs"
LOG_FILE          = LOGS_DIR / "sentinel.log"
FLAGGED_LOG       = LOGS_DIR / "flagged_transactions.jsonl"

# Preprocessing
TEST_SIZE         = 0.20
RANDOM_STATE      = 42
AMOUNT_LOG_TRANSFORM = True 

# Class Imbalance Handling
IMBALANCE_STRATEGY = "smote"
SMOTE_SAMPLING_RATIO = 0.3

# XGBoost Hyperparameters
XGBOOST_PARAMS = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma":            1,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "scale_pos_weight": 10,  
    "eval_metric":      "aucpr",  
    "use_label_encoder": False,
    "random_state":     RANDOM_STATE,
    "n_jobs":           -1,
}

# Isolation Forest
ISOLATION_FOREST_PARAMS = {
    "n_estimators":  300,
    "contamination": 0.002,
    "max_samples":   "auto",
    "random_state":  RANDOM_STATE,
    "n_jobs":        -1,
}

# Risk Scoring
ALPHA = 0.70   # weight for XGBoost probability
BETA  = 0.30   # weight for Isolation Forest anomaly score

# Risk thresholds
RISK_THRESHOLDS = {
    "low":      0.30,   # CLEAR
    "medium":   0.55,   # REVIEW
    "high":     0.75,   # FLAG
    # above 0.75        BLOCK
}

# API
API_HOST = "0.0.0.0"
API_PORT = 8000
API_RELOAD = True 

# Feature Engineering
HOUR_BINS = [0, 6, 12, 18, 24]
HOUR_LABELS = ["night", "morning", "afternoon", "evening"]
