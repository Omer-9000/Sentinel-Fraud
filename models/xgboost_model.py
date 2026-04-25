"""
models/xgboost_model.py — Primary Classifier for SentinelFraud

Why XGBoost?
• Handles class imbalance via scale_pos_weight
• Built-in regularisation (L1/L2) prevents overfitting on small fraud set
• Fast inference (< 1 ms per transaction at production scale)
• Native feature importance — feeds directly into SHAP explainability layer
• Industry-standard choice at major banks (JPMorgan, Stripe, Adyen all use GBMs)

Training strategy:
• Early stopping on a validation fold to prevent overfitting
• Precision-Recall AUC as primary metric (better than ROC-AUC for imbalanced data)
• Threshold tuning post-training to hit target false-positive rate
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RANDOM_STATE, XGBOOST_PARAMS, XGBOOST_PATH
from utils.logger import get_logger

log = get_logger("xgboost_model")


class FraudXGBoostClassifier:
    """
    Wrapper around XGBClassifier that adds:
    - Cross-validated training with early stopping
    - Threshold calibration to meet a target false-positive rate
    - Feature importance extraction
    """

    def __init__(self, params: dict = None):
        self.params = params or XGBOOST_PARAMS.copy()
        self.model: xgb.XGBClassifier | None = None
        self.optimal_threshold: float = 0.5
        self.feature_names: list[str] = []

    # Training
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "FraudXGBoostClassifier":
        """
        Train the XGBoost model with optional early stopping.
        If no validation set is supplied, a 10% internal split is used.
        """
        self.feature_names = list(X_train.columns)

        if X_val is None:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=0.10,
                stratify=y_train,
                random_state=RANDOM_STATE,
            )

        log.info(
            f"Training XGBoost on {len(X_train):,} samples "
            f"(val: {len(X_val):,}) …"
        )

        self.model = xgb.XGBClassifier(**self.params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        val_proba = self.model.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, val_proba)
        roc    = roc_auc_score(y_val, val_proba)
        log.info(f"Validation  PR-AUC: {pr_auc:.4f}  |  ROC-AUC: {roc:.4f}")

        # Calibrate threshold on the validation set
        self.optimal_threshold = self._calibrate_threshold(y_val, val_proba)
        log.info(f"Calibrated decision threshold: {self.optimal_threshold:.4f}")

        return self

    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5
    ) -> dict:
        """
        Stratified k-fold cross-validation — critical for imbalanced datasets.
        Returns mean and std of PR-AUC and ROC-AUC across folds.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        pr_aucs, roc_aucs = [], []

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

            m = xgb.XGBClassifier(**self.params)
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

            proba = m.predict_proba(X_val)[:, 1]
            pr_aucs.append(average_precision_score(y_val, proba))
            roc_aucs.append(roc_auc_score(y_val, proba))
            log.info(
                f"Fold {fold}/{n_splits}  PR-AUC: {pr_aucs[-1]:.4f}  "
                f"ROC-AUC: {roc_aucs[-1]:.4f}"
            )

        results = {
            "pr_auc_mean":  float(np.mean(pr_aucs)),
            "pr_auc_std":   float(np.std(pr_aucs)),
            "roc_auc_mean": float(np.mean(roc_aucs)),
            "roc_auc_std":  float(np.std(roc_aucs)),
        }
        log.info(
            f"CV Results → PR-AUC: {results['pr_auc_mean']:.4f} ± {results['pr_auc_std']:.4f}  |  "
            f"ROC-AUC: {results['roc_auc_mean']:.4f} ± {results['roc_auc_std']:.4f}"
        )
        return results

    # Inference
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns fraud probability for each row (shape: [n, 2])."""
        assert self.model is not None, "Model not trained yet."
        return self.model.predict_proba(X)

    def predict_fraud_score(self, X: pd.DataFrame) -> np.ndarray:
        """Returns fraud probability score (0–1) for each transaction."""
        return self.predict_proba(X)[:, 1]

    def predict_labels(self, X: pd.DataFrame) -> np.ndarray:
        """Returns binary labels using the calibrated threshold."""
        return (self.predict_fraud_score(X) >= self.optimal_threshold).astype(int)

    # Threshold Calibration
    def _calibrate_threshold(
        self, y_true: pd.Series, y_proba: np.ndarray, target_fpr: float = 0.01
    ) -> float:
        """
        Find the highest threshold that keeps the false-positive rate ≤ target_fpr.
        In banking, FPR control is critical — every false positive is a customer
        whose legitimate payment was declined (churn risk + operational cost).
        """
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)

        # Select thresholds where FPR ≤ target
        valid_mask = fpr <= target_fpr
        if not valid_mask.any():
            log.warning(
                f"Cannot achieve FPR ≤ {target_fpr:.2%}; using default threshold 0.5"
            )
            return 0.5

        # Among valid thresholds, pick the one with highest TPR
        best_idx = np.argmax(tpr[valid_mask])
        return float(thresholds[valid_mask][best_idx])

    # Feature Importance
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Returns a DataFrame of top-N features by XGBoost gain importance."""
        assert self.model is not None, "Model not trained yet."
        importance = self.model.get_booster().get_score(importance_type="gain")
        df = pd.DataFrame(
            importance.items(), columns=["feature", "importance"]
        ).sort_values("importance", ascending=False).head(top_n)
        return df.reset_index(drop=True)

    # Persistence
    def save(self, path: Path = XGBOOST_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        log.info(f"XGBoost model saved to {path}")

    @classmethod
    def load(cls, path: Path = XGBOOST_PATH) -> "FraudXGBoostClassifier":
        model = joblib.load(path)
        log.info(f"XGBoost model loaded from {path}")
        return model
