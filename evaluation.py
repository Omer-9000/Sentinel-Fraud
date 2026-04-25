"""
evaluation.py — Model Evaluation & Reporting for SentinelFraud

Banking-relevant metrics:
  • ROC-AUC:    Overall discrimination ability. Good baseline, but misleading
                when classes are very imbalanced (0.99 AUC ≠ good fraud detection).
  • PR-AUC:     Area under Precision-Recall curve. Superior metric for imbalanced
                problems — directly measures precision/recall trade-off on fraud class.
  • Precision:  Of transactions we flagged, what % were actually fraud?
                Low precision = too many false positives = angry customers.
  • Recall:     Of all actual frauds, what % did we catch?
                Low recall = missed fraud = direct financial loss.
  • F1-score:   Harmonic mean of precision & recall. Balanced single metric.
  • F2-score:   Weights recall 2x over precision. Banks often prefer catching
                more fraud even at cost of some false positives.
  • KS Stat:    Kolmogorov-Smirnov statistic — measures separation between fraud
                and legit score distributions. Standard metric in credit risk.
  • Brier Score: Calibration metric — how well-calibrated are the probabilities?
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import REPORTS_DIR
from utils.logger import get_logger

log = get_logger("evaluation")


# Core Metrics
def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute the full suite of fraud-relevant metrics.
    Returns a dictionary suitable for JSON serialisation.
    """
    y_pred = (y_proba >= threshold).astype(int)

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Precision, Recall, F1, F2
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = f1_score(y_true, y_pred, zero_division=0)
    f2        = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

    # AUC metrics
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc  = average_precision_score(y_true, y_proba)

    # False Positive Rate (critical for customer experience)
    fpr_at_threshold = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # KS Statistic (separation between fraud and legit distributions)
    fpr_arr, tpr_arr, _ = roc_curve(y_true, y_proba)
    ks_stat = float(np.max(tpr_arr - fpr_arr))

    # Brier Score (probability calibration — lower is better)
    brier = brier_score_loss(y_true, y_proba)

    metrics = {
        "threshold":         round(threshold, 4),
        "roc_auc":           round(roc_auc, 4),
        "pr_auc":            round(pr_auc, 4),
        "precision":         round(precision, 4),
        "recall":            round(recall, 4),
        "f1_score":          round(f1, 4),
        "f2_score":          round(f2, 4),
        "ks_statistic":      round(ks_stat, 4),
        "brier_score":       round(brier, 4),
        "fpr_at_threshold":  round(fpr_at_threshold, 4),
        "true_positives":    int(tp),
        "false_positives":   int(fp),
        "true_negatives":    int(tn),
        "false_negatives":   int(fn),
        "total_fraud":       int(tp + fn),
        "fraud_caught_pct":  round(recall * 100, 2),
        "false_alert_pct":   round(fpr_at_threshold * 100, 2),
    }

    _log_metrics(metrics)
    return metrics


def _log_metrics(m: dict) -> None:
    log.info(
        f"\n{'='*60}\n"
        f"  ROC-AUC:      {m['roc_auc']:.4f}\n"
        f"  PR-AUC:       {m['pr_auc']:.4f}\n"
        f"  Precision:    {m['precision']:.4f}\n"
        f"  Recall:       {m['recall']:.4f}  ({m['fraud_caught_pct']:.1f}% fraud caught)\n"
        f"  F1-Score:     {m['f1_score']:.4f}\n"
        f"  F2-Score:     {m['f2_score']:.4f}\n"
        f"  KS Statistic: {m['ks_statistic']:.4f}\n"
        f"  Brier Score:  {m['brier_score']:.4f}\n"
        f"  False Alerts: {m['false_alert_pct']:.2f}% of legit transactions flagged\n"
        f"  TP={m['true_positives']} FP={m['false_positives']} "
        f"TN={m['true_negatives']} FN={m['false_negatives']}\n"
        f"{'='*60}"
    )


# Plots
    y_true, y_proba, model_name: str = "SentinelFraud"
) -> Path:
    """Side-by-side ROC and Precision-Recall curves."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#0f1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1a1d2e")
        ax.tick_params(colors="#c0c0c0")
        ax.spines[:].set_color("#2a2d3e")

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color="#6c63ff", lw=2.5, label=f"ROC-AUC = {roc_auc:.4f}")
    ax1.plot([0, 1], [0, 1], color="#555", lw=1, linestyle="--", label="Random")
    ax1.fill_between(fpr, tpr, alpha=0.15, color="#6c63ff")
    ax1.set_xlabel("False Positive Rate", color="#c0c0c0")
    ax1.set_ylabel("True Positive Rate", color="#c0c0c0")
    ax1.set_title("ROC Curve", color="white", fontsize=13, pad=10)
    ax1.legend(facecolor="#1a1d2e", labelcolor="white", fontsize=10)

    # Precision, Recall
    precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall_arr, precision_arr)
    baseline = y_true.mean() if hasattr(y_true, "mean") else np.mean(y_true)
    ax2.plot(recall_arr, precision_arr, color="#00d4aa", lw=2.5,
             label=f"PR-AUC = {pr_auc:.4f}")
    ax2.axhline(baseline, color="#ff6b6b", lw=1.5, linestyle="--",
                label=f"Baseline (fraud rate {baseline:.3%})")
    ax2.fill_between(recall_arr, precision_arr, alpha=0.15, color="#00d4aa")
    ax2.set_xlabel("Recall", color="#c0c0c0")
    ax2.set_ylabel("Precision", color="#c0c0c0")
    ax2.set_title("Precision-Recall Curve", color="white", fontsize=13, pad=10)
    ax2.legend(facecolor="#1a1d2e", labelcolor="white", fontsize=10)

    plt.suptitle(f"{model_name} — Model Evaluation", color="white", fontsize=15, y=1.02)
    plt.tight_layout()

    out = REPORTS_DIR / "roc_pr_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    log.info(f"ROC/PR curves saved to {out}")
    return out


def plot_score_distribution(
    y_true, y_proba, model_name: str = "SentinelFraud"
) -> Path:
    """
    Distribution of risk scores split by actual class.
    A well-calibrated model shows clear separation between fraud and legit peaks.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d2e")
    ax.tick_params(colors="#c0c0c0")
    ax.spines[:].set_color("#2a2d3e")

    scores = np.array(y_proba)
    labels = np.array(y_true)

    ax.hist(scores[labels == 0], bins=80, alpha=0.7, color="#6c63ff",
            label="Legitimate", density=True)
    ax.hist(scores[labels == 1], bins=80, alpha=0.8, color="#ff6b6b",
            label="Fraud", density=True)

    # Threshold lines
    from config import RISK_THRESHOLDS
    colors_t = ["#ffd700", "#ff9900", "#ff4444"]
    labels_t = ["Low/Medium", "Medium/High", "High/Block"]
    for t_val, c, l in zip(RISK_THRESHOLDS.values(), colors_t, labels_t):
        ax.axvline(t_val, color=c, linestyle="--", lw=1.5, label=f"Threshold: {l} ({t_val})")

    ax.set_xlabel("Risk Score", color="#c0c0c0")
    ax.set_ylabel("Density", color="#c0c0c0")
    ax.set_title(f"{model_name} — Risk Score Distribution", color="white", fontsize=13)
    ax.legend(facecolor="#1a1d2e", labelcolor="white", fontsize=9)
    plt.tight_layout()

    out = REPORTS_DIR / "score_distribution.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    log.info(f"Score distribution plot saved to {out}")
    return out


def plot_confusion_matrix(y_true, y_pred, save_name: str = "confusion_matrix") -> Path:
    """Styled confusion matrix heatmap."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    labels = [["TN\n(Legit → CLEAR)", "FP\n(Legit → FLAGGED)"],
               ["FN\n(Fraud → MISSED)", "TP\n(Fraud → CAUGHT)"]]

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d2e")

    im = ax.imshow(cm, cmap="plasma", aspect="auto")
    plt.colorbar(im, ax=ax)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n{labels[i][j]}",
                    ha="center", va="center", color="white", fontsize=11)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted: Legit", "Predicted: Fraud"], color="#c0c0c0")
    ax.set_yticklabels(["Actual: Legit", "Actual: Fraud"], color="#c0c0c0")
    ax.set_title("Confusion Matrix", color="white", fontsize=13, pad=12)
    plt.tight_layout()

    out = REPORTS_DIR / f"{save_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    plt.close()
    log.info(f"Confusion matrix saved to {out}")
    return out


def generate_full_report(
    y_true,
    y_proba,
    threshold: float = 0.5,
    model_name: str = "SentinelFraud XGBoost+IF Hybrid",
) -> dict:
    """
    Master function: compute all metrics + generate all plots.
    Returns metrics dict and list of plot paths.
    """
    import json

    metrics = compute_metrics(y_true, y_proba, threshold)

    y_pred = (y_proba >= threshold).astype(int)

    plots = [
        str(plot_roc_pr_curves(y_true, y_proba, model_name)),
        str(plot_score_distribution(y_true, y_proba, model_name)),
        str(plot_confusion_matrix(y_true, y_pred)),
    ]

    # Persist metrics as JSON
    metrics_path = REPORTS_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Evaluation metrics saved to {metrics_path}")

    return {"metrics": metrics, "plots": plots, "metrics_path": str(metrics_path)}
