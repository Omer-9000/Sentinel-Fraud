"""
models/hybrid_scorer.py — Hybrid Risk Scoring Engine

The SentinelFraud risk score is a weighted blend of:
  • XGBoost fraud probability  (supervised — catches known fraud patterns)
  • Isolation Forest anomaly score (unsupervised — catches novel patterns)

  risk_score = α * xgb_prob + β * if_score
  (α=0.70, β=0.30 by default, configurable in config.py)

The final score maps to a four-tier decision:
  CLEAR  (<0.30)  → Pass through, no action
  REVIEW (0.30–0.55) → Queue for analyst review within 24 h
  FLAG   (0.55–0.75) → Trigger real-time alert, additional authentication
  BLOCK  (>0.75)  → Decline transaction, notify customer

This tiered approach is how modern banks (Mastercard, Visa, Revolut) actually
operate — binary FRAUD/NOT-FRAUD is too simplistic for real deployments.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import ALPHA, BETA, ISOLATION_FOREST_PATH, RISK_THRESHOLDS, XGBOOST_PATH
from models.anomaly_detector import FraudAnomalyDetector
from models.xgboost_model import FraudXGBoostClassifier
from utils.logger import get_logger

log = get_logger("hybrid_scorer")

RiskDecision = Literal["CLEAR", "REVIEW", "FLAG", "BLOCK"]


@dataclass
class ScoringResult:
    """Structured output from the hybrid scorer for a single transaction."""
    transaction_id: str
    xgb_score: float          # XGBoost fraud probability [0, 1]
    anomaly_score: float      # Isolation Forest normalised score [0, 1]
    risk_score: float         # Final blended score [0, 1]
    decision: RiskDecision    # Action tier
    is_novel_anomaly: bool    # True if IF flags as outlier but XGB disagrees
    explanation: dict         # SHAP values or feature importances


class HybridFraudScorer:
    """
    Loads trained models and produces a risk score + decision for each transaction.
    Designed for both batch scoring and real-time single-transaction inference.
    """

    def __init__(
        self,
        xgb_model: FraudXGBoostClassifier | None = None,
        anomaly_detector: FraudAnomalyDetector | None = None,
        alpha: float = ALPHA,
        beta: float = BETA,
    ):
        self.xgb = xgb_model
        self.if_det = anomaly_detector
        self.alpha = alpha
        self.beta = beta

    # Model Loading
    def load_models(
        self,
        xgb_path: Path = XGBOOST_PATH,
        if_path: Path = ISOLATION_FOREST_PATH,
    ) -> "HybridFraudScorer":
        self.xgb    = FraudXGBoostClassifier.load(xgb_path)
        self.if_det = FraudAnomalyDetector.load(if_path)
        log.info("Hybrid scorer: both models loaded successfully.")
        return self

    # Core Scoring
    def score(
        self, X: pd.DataFrame, transaction_ids: list[str] | None = None
    ) -> list[ScoringResult]:
        """
        Score a batch of preprocessed transactions.
        Returns a list of ScoringResult objects.
        """
        assert self.xgb is not None and self.if_det is not None, \
            "Call load_models() before scoring."

        xgb_probs    = self.xgb.predict_fraud_score(X)
        anom_scores  = self.if_det.anomaly_score(X)
        outlier_flags = self.if_det.predict_outlier(X)

        if transaction_ids is None:
            transaction_ids = [f"txn_{i:06d}" for i in range(len(X))]

        results = []
        for i, (xgb_s, anom_s, is_outlier, txn_id) in enumerate(
            zip(xgb_probs, anom_scores, outlier_flags, transaction_ids)
        ):
            hybrid = float(self.alpha * xgb_s + self.beta * anom_s)
            hybrid = float(np.clip(hybrid, 0.0, 1.0))
            decision = self._tier(hybrid)

            # Novel anomaly: IF flags it but XGB is relatively confident it's legit
            is_novel = bool(is_outlier and xgb_s < 0.30)

            results.append(
                ScoringResult(
                    transaction_id=txn_id,
                    xgb_score=round(float(xgb_s), 4),
                    anomaly_score=round(float(anom_s), 4),
                    risk_score=round(hybrid, 4),
                    decision=decision,
                    is_novel_anomaly=is_novel,
                    explanation={},  # populated separately by explainer
                )
            )

        flagged = sum(1 for r in results if r.decision in ("FLAG", "BLOCK"))
        log.info(
            f"Scored {len(results):,} transactions → "
            f"{flagged} flagged/blocked  "
            f"({flagged/len(results)*100:.2f}%)"
        )
        return results

    def score_single(
        self, X_row: pd.DataFrame, transaction_id: str = "txn_live"
    ) -> ScoringResult:
        """Convenience wrapper for single-transaction real-time scoring."""
        return self.score(X_row, transaction_ids=[transaction_id])[0]

    # Decision Tier
    @staticmethod
    def _tier(score: float) -> RiskDecision:
        t = RISK_THRESHOLDS
        if   score < t["low"]:    return "CLEAR"
        elif score < t["medium"]: return "REVIEW"
        elif score < t["high"]:   return "FLAG"
        else:                     return "BLOCK"

    # Batch DataFrame Output
    def score_to_dataframe(
        self, X: pd.DataFrame, transaction_ids: list[str] | None = None
    ) -> pd.DataFrame:
        """Scores a batch and returns results as a tidy DataFrame."""
        results = self.score(X, transaction_ids)
        rows = [
            {
                "transaction_id":  r.transaction_id,
                "xgb_score":       r.xgb_score,
                "anomaly_score":   r.anomaly_score,
                "risk_score":      r.risk_score,
                "decision":        r.decision,
                "is_novel_anomaly": r.is_novel_anomaly,
            }
            for r in results
        ]
        return pd.DataFrame(rows)
