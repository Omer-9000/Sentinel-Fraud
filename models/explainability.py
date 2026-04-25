"""
models/explainability.py — SHAP-Based Explainability for SentinelFraud

Why explainability matters in banking:
• Regulators (SR 11-7, EU AI Act) require model decisions to be auditable.
• Risk officers need to justify account actions to compliance teams.
• Customer service needs plain-English reasons to communicate a declined transaction.
• SHAP (SHapley Additive exPlanations) is the gold standard:
    - Model-agnostic, theoretically grounded (cooperative game theory)
    - Produces per-feature contributions for each individual prediction
    - Works natively with XGBoost (TreeExplainer — very fast)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import REPORTS_DIR
from utils.logger import get_logger

log = get_logger("explainability")


class FraudExplainer:
    """
    Wraps SHAP TreeExplainer to provide per-prediction and global explanations.
    """

    def __init__(self, xgb_model):
        """
        Parameters
        ----------
        xgb_model : FraudXGBoostClassifier
            A trained FraudXGBoostClassifier instance (not raw XGBClassifier).
        """
        assert xgb_model.model is not None, "XGBoost model must be trained first."
        log.info("Initialising SHAP TreeExplainer …")
        self.explainer = shap.TreeExplainer(xgb_model.model)
        self.feature_names = xgb_model.feature_names

    # Per-Transaction Explanation
    def explain_single(
        self, X_row: pd.DataFrame, top_n: int = 10
    ) -> dict:
        """
        Returns top-N SHAP feature contributions for a single transaction.
        Positive SHAP values push toward FRAUD; negative toward LEGIT.

        Example output:
            {
                "V14": -3.21,    # strongly indicates legit
                "Amount_log":  1.84,    # pushes toward fraud
                ...
                "base_value": 0.0042,  # expected model output (baseline)
            }
        """
        shap_values = self.explainer.shap_values(X_row)

        # For binary classification, shap_values may be a list [legit, fraud]
        if isinstance(shap_values, list):
            fraud_shap = shap_values[1][0]
        else:
            fraud_shap = shap_values[0]

        contribs = dict(zip(self.feature_names, fraud_shap.tolist()))

        # Keep top-N by absolute magnitude
        top = dict(
            sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        )
        top["base_value"] = float(self.explainer.expected_value
                                  if not isinstance(self.explainer.expected_value, list)
                                  else self.explainer.expected_value[1])

        return top

    def explain_in_plain_english(self, X_row: pd.DataFrame) -> str:
        """
        Generates a one-sentence human-readable explanation for customer service.
        """
        contribs = self.explain_single(X_row, top_n=3)
        contribs.pop("base_value", None)

        top_features = list(contribs.items())
        risk_drivers = [f for f, v in top_features if v > 0]
        safe_drivers = [f for f, v in top_features if v < 0]

        parts = []
        if risk_drivers:
            parts.append(f"Risk elevated by: {', '.join(risk_drivers)}")
        if safe_drivers:
            parts.append(f"Mitigated by: {', '.join(safe_drivers)}")

        return " | ".join(parts) if parts else "No dominant risk signals identified."

    # Global Feature Importance
    def global_feature_importance(
        self, X_sample: pd.DataFrame, max_display: int = 20
    ) -> pd.DataFrame:
        """
        Computes mean |SHAP value| per feature across a sample of transactions.
        This is more reliable than XGBoost's built-in gain importance.
        """
        log.info(
            f"Computing global SHAP importance on {len(X_sample):,} samples …"
        )
        shap_values = self.explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            fraud_shap = np.abs(shap_values[1])
        else:
            fraud_shap = np.abs(shap_values)

        mean_abs = fraud_shap.mean(axis=0)
        df = pd.DataFrame({
            "feature":    self.feature_names,
            "mean_shap":  mean_abs,
        }).sort_values("mean_shap", ascending=False).head(max_display)

        return df.reset_index(drop=True)

    # Visualisations
    def plot_summary(
        self, X_sample: pd.DataFrame, save_path: Path | None = None
    ) -> Path:
        """
        SHAP beeswarm summary plot — shows feature impact distribution across
        all sampled transactions. Standard deliverable for model audits.
        """
        shap_values = self.explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            sv = shap_values[1]
        else:
            sv = shap_values

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            sv, X_sample,
            feature_names=self.feature_names,
            show=False,
            max_display=20,
        )
        plt.title("SHAP Feature Impact — Fraud Detection Model", fontsize=14, pad=12)
        plt.tight_layout()

        if save_path is None:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            save_path = REPORTS_DIR / "shap_summary.png"

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"SHAP summary plot saved to {save_path}")
        return save_path

    def plot_waterfall_single(
        self, X_row: pd.DataFrame, transaction_id: str = "txn", save_path: Path | None = None
    ) -> Path:
        """
        Waterfall plot for a single transaction — shows each feature's
        push toward or away from fraud. Used in analyst dashboards.
        """
        shap_values = self.explainer(X_row)

        if hasattr(shap_values, "values") and len(shap_values.values.shape) == 3:
            # Multi-output: take fraud class
            import shap as _shap
            sv = _shap.Explanation(
                values=shap_values.values[0, :, 1],
                base_values=shap_values.base_values[0, 1],
                data=shap_values.data[0],
                feature_names=self.feature_names,
            )
        else:
            sv = shap_values[0]

        shap.plots.waterfall(sv, show=False)
        plt.title(f"SHAP Waterfall — {transaction_id}", fontsize=12)
        plt.tight_layout()

        if save_path is None:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            save_path = REPORTS_DIR / f"shap_waterfall_{transaction_id}.png"

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Waterfall plot saved to {save_path}")
        return save_path
