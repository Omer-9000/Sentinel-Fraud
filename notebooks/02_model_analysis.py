"""
notebooks/02_model_analysis.py — Post-training Model Analysis
Run after train.py to get deeper insights into model behaviour.

Sections:
  1. Load trained models
  2. Score holdout test set
  3. Threshold sweep — precision/recall trade-off
  4. Decision tier breakdown
  5. Novel anomaly analysis
  6. Confusion matrix at calibrated threshold
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import REPORTS_DIR, TEST_PATH
from models.hybrid_scorer import HybridFraudScorer
from evaluation import compute_metrics, plot_confusion_matrix

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading test data …")
test_df = pd.read_parquet(TEST_PATH)
X_test = test_df.drop(columns=["Class"])
y_test = test_df["Class"]

print("Loading models …")
scorer = HybridFraudScorer().load_models()
xgb_clf = scorer.xgb
if_det  = scorer.if_det

# Score full test set 
print("Scoring test set …")
score_df = scorer.score_to_dataframe(X_test)
score_df["true_label"] = y_test.values
xgb_scores = xgb_clf.predict_fraud_score(X_test)

# Calibrated metrics
print("\n── Calibrated Threshold Metrics ──────────────────────────────")
metrics = compute_metrics(y_test, score_df["risk_score"].values,
                          threshold=xgb_clf.optimal_threshold)
for k, v in metrics.items():
    print(f"  {k:<25} {v}")

# Threshold sweep 
thresholds = np.linspace(0.01, 0.99, 200)
precisions, recalls, f1s, fprs = [], [], [], []
for t in thresholds:
    m = compute_metrics(y_test.values, score_df["risk_score"].values, threshold=t)
    precisions.append(m["precision"]); recalls.append(m["recall"])
    f1s.append(m["f1_score"]); fprs.append(m["fpr_at_threshold"])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0f1117")
for ax in axes:
    ax.set_facecolor("#1a1d2e"); ax.tick_params(colors="#c0c0c0"); ax.spines[:].set_color("#2a2d3e")

axes[0].plot(thresholds, precisions, color="#6c63ff", lw=2, label="Precision")
axes[0].plot(thresholds, recalls,    color="#ff6b6b", lw=2, label="Recall")
axes[0].plot(thresholds, f1s,        color="#00d4aa", lw=2, label="F1")
axes[0].axvline(xgb_clf.optimal_threshold, color="#ffd700", ls="--", lw=1.5,
                label=f"Calibrated ({xgb_clf.optimal_threshold:.2f})")
axes[0].set_xlabel("Threshold", color="#c0c0c0")
axes[0].set_title("Precision / Recall / F1 vs Threshold", color="white")
axes[0].legend(facecolor="#1a1d2e", labelcolor="white")

axes[1].plot(thresholds, [f*100 for f in fprs], color="#ff9900", lw=2)
axes[1].axhline(1.0, color="#ff4444", ls="--", lw=1.5, label="1% FPR target")
axes[1].axvline(xgb_clf.optimal_threshold, color="#ffd700", ls="--", lw=1.5,
                label=f"Calibrated ({xgb_clf.optimal_threshold:.2f})")
axes[1].set_xlabel("Threshold", color="#c0c0c0")
axes[1].set_ylabel("False Positive Rate (%)", color="#c0c0c0")
axes[1].set_title("False Positive Rate vs Threshold", color="white")
axes[1].legend(facecolor="#1a1d2e", labelcolor="white")

plt.tight_layout()
out = REPORTS_DIR / "model_threshold_sweep.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f1117"); plt.close()
print(f"\nSaved threshold sweep: {out}")

# Decision tier breakdown 
tier_counts = score_df.groupby(["decision", "true_label"]).size().unstack(fill_value=0)
print("\n── Decision Tier Breakdown ───────────────────────────────────")
print(tier_counts.to_string())

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#1a1d2e")
ax.tick_params(colors="#c0c0c0"); ax.spines[:].set_color("#2a2d3e")
tier_order = ["CLEAR", "REVIEW", "FLAG", "BLOCK"]
tier_counts_ordered = tier_counts.reindex(tier_order, fill_value=0)
x = np.arange(len(tier_order))
w = 0.35
if 0 in tier_counts_ordered.columns:
    ax.bar(x - w/2, tier_counts_ordered[0], w, label="Legitimate", color="#6c63ff", alpha=0.85)
if 1 in tier_counts_ordered.columns:
    ax.bar(x + w/2, tier_counts_ordered[1], w, label="Fraud",      color="#ff6b6b", alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(tier_order, color="#c0c0c0")
ax.set_title("Decision Tier Distribution by True Class", color="white", fontsize=13)
ax.set_ylabel("Count", color="#c0c0c0")
ax.legend(facecolor="#1a1d2e", labelcolor="white")
plt.tight_layout()
out = REPORTS_DIR / "model_decision_tiers.png"
plt.savefig(out, dpi=150, facecolor="#0f1117"); plt.close()
print(f"Saved tier breakdown: {out}")

# Novel anomalies 
novel = score_df[(score_df["is_novel_anomaly"]) & (score_df["true_label"] == 1)]
print(f"\n── Novel Anomalies ───────────────────────────────────────────")
print(f"  Fraud txns flagged by IF but low XGB score: {len(novel)}")
if len(novel) > 0:
    print(novel[["risk_score", "xgb_score", "anomaly_score", "decision"]].describe().to_string())

# Confusion matrix 
y_pred = (score_df["risk_score"] >= xgb_clf.optimal_threshold).astype(int)
plot_confusion_matrix(y_test.values, y_pred.values, save_name="model_confusion_matrix")
print(f"\n✅ Model analysis complete — all plots saved to reports/")
