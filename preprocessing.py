"""
preprocessing.py — Feature Engineering & Data Preparation for SentinelFraud

Pipeline steps (in order):
  1. Load raw CSV
  2. Remove exact duplicates
  3. Log-transform Amount (reduces right-skew)
  4. Cyclical encode Time (hour-of-day sine/cosine)
  5. Engineer velocity & ratio features
  6. Scale continuous features with RobustScaler
  7. Split into stratified train / test sets
  8. Apply SMOTE to training set only (never to test!)
  9. Persist processed artifacts to disk
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    FEATURE_LIST_PATH,
    IMBALANCE_STRATEGY,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    SCALER_PATH,
    SMOTE_SAMPLING_RATIO,
    TEST_PATH,
    TEST_SIZE,
    TRAIN_PATH,
)
from utils.logger import get_logger

log = get_logger("preprocessing")


# Load
def load_raw(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    log.info(f"Loading raw data from {path} …")
    df = pd.read_csv(path)
    log.info(f"Loaded {len(df):,} rows  |  fraud rate: {df['Class'].mean()*100:.3f}%")
    return df


# Clean
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df = df.drop_duplicates()
    log.info(f"Removed {n_before - len(df):,} duplicate rows.")
    return df


# Core Transformations
def transform_amount(df: pd.DataFrame) -> pd.DataFrame:
    """log1p stabilises variance and reduces influence of high-value outliers."""
    df = df.copy()
    df["Amount_log"] = np.log1p(df["Amount"])
    df.drop(columns=["Amount"], inplace=True)
    return df


def encode_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw Time (seconds) into hour-of-day, then encode cyclically.
    Cyclical encoding (sin/cos) ensures 23:59 is 'close' to 00:00 in feature space.
    """
    df = df.copy()
    seconds_in_day = 24 * 3600
    hour_of_day = (df["Time"] % seconds_in_day) / 3600  # 0–24
    df["Time_sin"] = np.sin(2 * np.pi * hour_of_day / 24)
    df["Time_cos"] = np.cos(2 * np.pi * hour_of_day / 24)
    df.drop(columns=["Time"], inplace=True)
    return df


# Feature Engineering
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Domain-driven features derived from PCA components and Amount.
    These are the kinds of features a fraud analyst would hand-craft:

    • v_pca_norm   — L2 norm of all V-features (overall statistical deviance)
    • v_high_risk  — sum of high-importance V-features (empirically chosen)
    • amount_v_ratio — Amount relative to dominant PCA direction
    • amount_v17_interaction — V17 is strongly negative for fraud in real data
    """
    df = df.copy()
    v_cols = [f"V{i}" for i in range(1, 29)]

    # Magnitude of the full PCA vector → how "unusual" the transaction pattern is
    df["v_pca_norm"] = np.linalg.norm(df[v_cols].values, axis=1)

    # High-importance PCA features (identified via prior SHAP analysis on real data)
    high_risk_cols = ["V4", "V11", "V12", "V14", "V16", "V17", "V18"]
    df["v_high_risk_sum"] = df[high_risk_cols].sum(axis=1)

    # Amount vs. primary PCA direction interaction
    df["amount_v_ratio"] = df["Amount_log"] / (np.abs(df["V1"]) + 1e-8)

    # V17 interaction (fraud transactions have large negative V17)
    df["amount_v17_interaction"] = df["Amount_log"] * df["V17"]

    # Outlier flag: any single V-feature beyond 3σ?
    v_abs_max = df[v_cols].abs().max(axis=1)
    df["extreme_pca_flag"] = (v_abs_max > 20).astype(int)

    log.info(
        f"Engineered 5 new features: v_pca_norm, v_high_risk_sum, "
        f"amount_v_ratio, amount_v17_interaction, extreme_pca_flag"
    )
    return df


# Scaling
def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scale_cols: list[str],
    fit: bool = True,
    scaler_path: Path = SCALER_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    RobustScaler is used instead of StandardScaler because it is less
    influenced by outliers — common in financial transactions.
    Fit ONLY on training data; transform both train and test.
    """
    if fit:
        scaler = RobustScaler()
        X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
        joblib.dump(scaler, scaler_path)
        log.info(f"Scaler fitted and saved to {scaler_path}")
    else:
        scaler = joblib.load(scaler_path)
        log.info(f"Scaler loaded from {scaler_path}")
        X_train[scale_cols] = scaler.transform(X_train[scale_cols])

    X_test[scale_cols] = scaler.transform(X_test[scale_cols])
    return X_train, X_test


# Train / Test Split
def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=["Class"])
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    log.info(
        f"Train: {len(X_train):,} rows ({y_train.sum():,} fraud)  |  "
        f"Test:  {len(X_test):,} rows ({y_test.sum():,} fraud)"
    )
    return X_train, X_test, y_train, y_test


# SMOTE
def apply_smote(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """
    SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic
    fraud examples by interpolating between existing ones.
    Applied ONLY to training data — applying to test would be data leakage.
    """
    if IMBALANCE_STRATEGY not in ("smote", "both"):
        log.info("SMOTE disabled — using class_weight strategy only.")
        return X_train, y_train

    smote = SMOTE(
        sampling_strategy=SMOTE_SAMPLING_RATIO,
        random_state=RANDOM_STATE,
        k_neighbors=5,
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)
    log.info(
        f"After SMOTE: {len(X_res):,} rows  "
        f"({int(y_res.sum()):,} fraud = {y_res.mean()*100:.1f}%)"
    )
    return pd.DataFrame(X_res, columns=X_train.columns), pd.Series(y_res, name="Class")


# Master Pipeline
def run_preprocessing_pipeline() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
]:
    """
    End-to-end preprocessing. Returns ready-to-train splits.
    Also persists train/test parquet files for reproducibility.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    df = remove_duplicates(df)
    df = transform_amount(df)
    df = encode_time(df)
    df = engineer_features(df)

    X_train, X_test, y_train, y_test = split_data(df)

    # Determine which columns need scaling
    scale_cols = ["Amount_log", "v_pca_norm", "v_high_risk_sum",
                  "amount_v_ratio", "amount_v17_interaction"]
    scale_cols = [c for c in scale_cols if c in X_train.columns]

    X_train, X_test = scale_features(X_train, X_test, scale_cols)

    # Persist feature list (used at inference time to enforce column order)
    feature_columns = list(X_train.columns)
    joblib.dump(feature_columns, FEATURE_LIST_PATH)
    log.info(f"Feature list ({len(feature_columns)} columns) saved to {FEATURE_LIST_PATH}")

    # Apply SMOTE to training data only
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)

    # Persist to parquet for auditability
    train_df = X_train_bal.copy()
    train_df["Class"] = y_train_bal.values
    train_df.to_parquet(TRAIN_PATH, index=False)

    test_df = X_test.copy()
    test_df["Class"] = y_test.values
    test_df.to_parquet(TEST_PATH, index=False)
    log.info(f"Processed datasets saved to {PROCESSED_DATA_DIR}")

    return X_train_bal, X_test, y_train_bal, y_test


# ── Inference-time preprocessing (single transaction)
def preprocess_single(raw: dict) -> pd.DataFrame:
    """
    Apply the identical transformations used during training to a single
    incoming transaction dict (used by the API at inference time).
    """
    df = pd.DataFrame([raw])

    # Same transformations as training
    df["Amount_log"] = np.log1p(df["Amount"])
    df.drop(columns=["Amount"], inplace=True, errors="ignore")

    seconds_in_day = 24 * 3600
    hour_of_day = (df["Time"] % seconds_in_day) / 3600
    df["Time_sin"] = np.sin(2 * np.pi * hour_of_day / 24)
    df["Time_cos"] = np.cos(2 * np.pi * hour_of_day / 24)
    df.drop(columns=["Time"], inplace=True, errors="ignore")

    v_cols = [f"V{i}" for i in range(1, 29)]
    existing_v = [c for c in v_cols if c in df.columns]
    df["v_pca_norm"] = np.linalg.norm(df[existing_v].values, axis=1)

    high_risk_cols = [c for c in ["V4", "V11", "V12", "V14", "V16", "V17", "V18"] if c in df.columns]
    df["v_high_risk_sum"] = df[high_risk_cols].sum(axis=1)
    df["amount_v_ratio"] = df["Amount_log"] / (np.abs(df.get("V1", pd.Series([1.0]))) + 1e-8)
    df["amount_v17_interaction"] = df["Amount_log"] * df.get("V17", pd.Series([0.0]))

    v_abs_max = df[existing_v].abs().max(axis=1)
    df["extreme_pca_flag"] = (v_abs_max > 20).astype(int)

    # Load scaler and apply
    scaler: RobustScaler = joblib.load(SCALER_PATH)
    feature_columns: list = joblib.load(FEATURE_LIST_PATH)

    scale_cols = ["Amount_log", "v_pca_norm", "v_high_risk_sum",
                  "amount_v_ratio", "amount_v17_interaction"]
    scale_cols = [c for c in scale_cols if c in df.columns]
    df[scale_cols] = scaler.transform(df[scale_cols])

    # Enforce column order and fill any missing columns with 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0.0
    return df[feature_columns]


if __name__ == "__main__":
    run_preprocessing_pipeline()
