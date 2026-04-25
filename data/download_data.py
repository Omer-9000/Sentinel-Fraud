"""
data/download_data.py — Dataset Acquisition for SentinelFraud
Automatically downloads the Kaggle Credit Card Fraud dataset.

The dataset contains 284,807 transactions (492 frauds = 0.172%).
Features V1–V28 are PCA-transformed (anonymised) bank features.
'Amount' and 'Time' are the only raw features.
'Class' = 1 → fraud, 0 → legitimate.

Usage:
    python data/download_data.py
    # or, if Kaggle CLI is not set up:
    python data/download_data.py --synthetic
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

# Allow running this file directly from its own directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW_DATA_DIR, RAW_DATA_PATH
from utils.logger import get_logger

log = get_logger("data_download")


def download_via_kaggle() -> None:
    """Download the dataset using the official Kaggle CLI."""
    log.info("Downloading creditcard.csv via Kaggle API …")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", "mlg-ulb/creditcardfraud",
                "-p", str(RAW_DATA_DIR),
                "--unzip",
            ],
            check=True,
        )
        log.info(f"Dataset saved to {RAW_DATA_PATH}")
    except FileNotFoundError:
        log.error(
            "Kaggle CLI not found. Install it with: pip install kaggle\n"
            "Then place your kaggle.json in ~/.kaggle/ and retry.\n"
            "Alternatively, run: python data/download_data.py --synthetic"
        )
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        log.error(f"Kaggle download failed: {exc}")
        sys.exit(1)


def generate_synthetic_dataset(n_legit: int = 284_315, n_fraud: int = 492) -> None:
    """
    Generate a statistically realistic synthetic dataset that mirrors the
    structure of the real Kaggle creditcardfraud dataset.
    Useful for offline development / CI pipelines.
    """
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_classification

    log.info(
        f"Generating synthetic dataset: {n_legit} legit + {n_fraud} fraud transactions …"
    )
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    n_total = n_legit + n_fraud

    # V1–V28: correlated Gaussian features (simulates PCA-transformed bank data)
    X, y = make_classification(
        n_samples=n_total,
        n_features=28,
        n_informative=20,
        n_redundant=5,
        n_clusters_per_class=2,
        weights=[n_legit / n_total, n_fraud / n_total],
        flip_y=0.001,
        random_state=42,
    )

    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])

    # Time: seconds elapsed (0 – 172,792 s ≈ 2 days)
    df["Time"] = np.sort(rng.uniform(0, 172_792, n_total))

    # Amount: log-normal distribution similar to the real dataset
    # Fraudulent amounts tend to be smaller to avoid detection
    amounts = np.where(
        y == 0,
        rng.lognormal(mean=3.5, sigma=2.0, size=n_total),   # legit: wider range
        rng.lognormal(mean=2.0, sigma=1.5, size=n_total),   # fraud: smaller
    )
    df["Amount"] = np.clip(amounts, 0.01, 25_691.16)
    df["Class"]  = y

    df.to_csv(RAW_DATA_PATH, index=False)
    log.info(
        f"Synthetic dataset written to {RAW_DATA_PATH}  "
        f"({n_fraud} fraud / {n_total} total = {n_fraud/n_total*100:.3f}%)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acquire fraud detection dataset")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate a synthetic dataset instead of downloading from Kaggle.",
    )
    args = parser.parse_args()

    if RAW_DATA_PATH.exists():
        log.info(f"Dataset already exists at {RAW_DATA_PATH} — skipping download.")
    elif args.synthetic:
        generate_synthetic_dataset()
    else:
        download_via_kaggle()
