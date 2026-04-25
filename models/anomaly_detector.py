"""
models/anomaly_detector.py — Isolation Forest Anomaly Component

Why add anomaly detection alongside XGBoost?
• XGBoost is a supervised model — it only detects fraud patterns it was trained on.
• Novel fraud schemes (zero-day fraud) look "normal" to the classifier.
• Isolation Forest is unsupervised — flags statistical outliers regardless of label.
• Combining both gives a HYBRID system: catches known AND unknown fraud patterns.

Isolation Forest intuition:
• Randomly partition the feature space using decision trees.
• Anomalies require FEWER splits to isolate (they're in sparse regions).
• anomaly_score = −mean_path_length (higher = more anomalous).
• We normalise this to [0, 1] and weight it in the final risk score.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ISOLATION_FOREST_PARAMS, ISOLATION_FOREST_PATH
from utils.logger import get_logger

log = get_logger("anomaly_detector")


class FraudAnomalyDetector:
    """
    Isolation Forest wrapper that exposes a normalised anomaly score in [0, 1].
    Higher score = more anomalous = more likely to be fraud.
    """

    def __init__(self, params: dict = None):
        self.params = params or ISOLATION_FOREST_PARAMS.copy()
        self.model: IsolationForest | None = None
        self._score_min: float = -1.0
        self._score_max: float = 1.0

    # Training
    def train(self, X_train: pd.DataFrame) -> "FraudAnomalyDetector":
        """
        Isolation Forest is trained on the FULL training set (both classes).
        The model learns the normal distribution and flags deviations.
        Note: Training only on legit transactions is an alternative strategy
        — useful when fraud examples are completely unavailable.
        """
        log.info(
            f"Training Isolation Forest on {len(X_train):,} samples "
            f"(contamination={self.params['contamination']}) …"
        )
        self.model = IsolationForest(**self.params)
        self.model.fit(X_train)

        # Compute raw score bounds on training data for normalisation
        raw_scores = self.model.score_samples(X_train)
        self._score_min = float(raw_scores.min())
        self._score_max = float(raw_scores.max())

        log.info(
            f"Isolation Forest trained.  "
            f"Raw score range: [{self._score_min:.4f}, {self._score_max:.4f}]"
        )
        return self

    # Inference
    def anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns a normalised anomaly score in [0, 1] for each transaction.
        0 = very normal  |  1 = extreme outlier

        Isolation Forest's score_samples() returns negative average path lengths.
        Lower (more negative) = more anomalous. We invert and normalise.
        """
        assert self.model is not None, "Detector not trained yet."
        raw = self.model.score_samples(X)
        # Invert: lower raw score → higher anomaly
        inverted = -raw
        # Normalise using training bounds
        norm_min = -self._score_max
        norm_max = -self._score_min
        denom = norm_max - norm_min if norm_max != norm_min else 1.0
        normalised = np.clip((inverted - norm_min) / denom, 0.0, 1.0)
        return normalised

    def predict_outlier(self, X: pd.DataFrame) -> np.ndarray:
        """Returns 1 for outlier (potential novel fraud), 0 for normal."""
        assert self.model is not None, "Detector not trained yet."
        preds = self.model.predict(X)
        return (preds == -1).astype(int)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluate anomaly detection against ground-truth labels.
        Metrics: ROC-AUC, PR-AUC, outlier detection recall on fraud class.
        """
        from sklearn.metrics import average_precision_score, roc_auc_score

        scores = self.anomaly_score(X_test)
        outlier_labels = self.predict_outlier(X_test)

        roc   = roc_auc_score(y_test, scores)
        pr    = average_precision_score(y_test, scores)
        fraud_recall = float(
            outlier_labels[y_test == 1].mean()
        ) if y_test.sum() > 0 else 0.0

        results = {
            "if_roc_auc":      round(roc, 4),
            "if_pr_auc":       round(pr, 4),
            "if_fraud_recall": round(fraud_recall, 4),
        }
        log.info(
            f"Isolation Forest Eval → ROC-AUC: {roc:.4f}  |  "
            f"PR-AUC: {pr:.4f}  |  Fraud recall: {fraud_recall:.4f}"
        )
        return results

    # Persistence
    def save(self, path: Path = ISOLATION_FOREST_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        log.info(f"Anomaly detector saved to {path}")

    @classmethod
    def load(cls, path: Path = ISOLATION_FOREST_PATH) -> "FraudAnomalyDetector":
        det = joblib.load(path)
        log.info(f"Anomaly detector loaded from {path}")
        return det
