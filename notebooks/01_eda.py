"""
notebooks/01_eda.py — Exploratory Data Analysis for SentinelFraud
Run as a plain Python script or paste into a Jupyter notebook cell-by-cell.

Sections:
  1. Dataset overview and class distribution
  2. Feature correlation with fraud label
  3. Amount and Time distributions by class
  4. PCA feature distributions (V1–V28 boxplots)
  5. Correlation heatmap
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW_DATA_PATH, REPORTS_DIR

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="darkgrid", palette="muted")

print("Loading data …")
df = pd.read_csv(RAW_DATA_PATH)

# Dataset Overview
print("\n── Dataset Overview ──────────────────────────────────────────")
print(df.shape)
print(df.dtypes)
print(df.isnull().sum().sum(), "missing values")
print(df.duplicated().sum(), "duplicate rows")
print("\nClass distribution:")
vc = df["Class"].value_counts()
print(vc)
print(f"Fraud rate: {df['Class'].mean()*100:.4f}%")

# Class Imbalance Bar
fig, ax = plt.subplots(figsize=(7, 4))
fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#1a1d2e")
bars = ax.bar(["Legitimate", "Fraud"], vc.values, color=["#6c63ff", "#ff6b6b"], width=0.5)
ax.bar_label(bars, fmt="{:,.0f}", color="white", padding=4)
ax.set_title("Transaction Class Distribution", color="white", fontsize=13)
ax.tick_params(colors="#c0c0c0")
ax.spines[:].set_color("#2a2d3e")
ax.set_yscale("log")
ax.set_ylabel("Count (log scale)", color="#c0c0c0")
plt.tight_layout()
out = REPORTS_DIR / "eda_class_distribution.png"
plt.savefig(out, dpi=150, facecolor="#0f1117"); plt.close()
print(f"\nSaved: {out}")

# Amount Distribution by Class
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor("#0f1117")
for ax in axes: ax.set_facecolor("#1a1d2e"); ax.tick_params(colors="#c0c0c0"); ax.spines[:].set_color("#2a2d3e")

legit = df[df["Class"] == 0]["Amount"]
fraud = df[df["Class"] == 1]["Amount"]

axes[0].hist(legit, bins=100, color="#6c63ff", alpha=0.8, density=True)
axes[0].set_title("Legitimate: Amount Distribution", color="white")
axes[0].set_xlabel("Amount ($)", color="#c0c0c0")

axes[1].hist(fraud, bins=50, color="#ff6b6b", alpha=0.8, density=True)
axes[1].set_title("Fraud: Amount Distribution", color="white")
axes[1].set_xlabel("Amount ($)", color="#c0c0c0")

plt.suptitle("Transaction Amount by Class", color="white", fontsize=14, y=1.02)
plt.tight_layout()
out = REPORTS_DIR / "eda_amount_distribution.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f1117"); plt.close()
print(f"Saved: {out}")

# Correlation of V-features with Class
v_cols = [f"V{i}" for i in range(1, 29)]
correlations = df[v_cols + ["Amount", "Time", "Class"]].corr()["Class"].drop("Class").sort_values()

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor("#0f1117"); ax.set_facecolor("#1a1d2e")
colors = ["#ff6b6b" if v > 0 else "#6c63ff" for v in correlations.values]
ax.barh(correlations.index, correlations.values, color=colors)
ax.axvline(0, color="white", lw=0.8)
ax.set_title("Feature Correlation with Fraud Label", color="white", fontsize=13)
ax.tick_params(colors="#c0c0c0"); ax.spines[:].set_color("#2a2d3e")
ax.set_xlabel("Pearson Correlation", color="#c0c0c0")
plt.tight_layout()
out = REPORTS_DIR / "eda_feature_correlations.png"
plt.savefig(out, dpi=150, facecolor="#0f1117"); plt.close()
print(f"Saved: {out}")

# Top discriminating V-features boxplots
top_features = correlations.abs().sort_values(ascending=False).head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.patch.set_facecolor("#0f1117")
axes_flat = axes.flatten()
for i, feat in enumerate(top_features):
    ax = axes_flat[i]
    ax.set_facecolor("#1a1d2e"); ax.tick_params(colors="#c0c0c0"); ax.spines[:].set_color("#2a2d3e")
    data = [df[df["Class"] == 0][feat].values, df[df["Class"] == 1][feat].values]
    bp = ax.boxplot(data, patch_artist=True,
                    boxprops=dict(facecolor="#6c63ff", color="white"),
                    medianprops=dict(color="#00d4aa", lw=2),
                    whiskerprops=dict(color="white"),
                    capprops=dict(color="white"),
                    flierprops=dict(marker=".", color="#ff6b6b", alpha=0.3, markersize=2))
    bp["boxes"][1].set_facecolor("#ff6b6b")
    ax.set_xticklabels(["Legitimate", "Fraud"], color="#c0c0c0")
    ax.set_title(feat, color="white")

plt.suptitle("Top Discriminating PCA Features by Class", color="white", fontsize=14, y=1.02)
plt.tight_layout()
out = REPORTS_DIR / "eda_top_features_boxplot.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0f1117"); plt.close()
print(f"Saved: {out}")

print("\n✅ EDA complete — all plots saved to reports/")
