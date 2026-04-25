"""
utils/monitoring.py — Runtime Monitoring Simulation for SentinelFraud

In production, fraud models degrade silently as fraud patterns evolve.
This module simulates the monitoring layer that a real MLOps team would build:

  • Score distribution drift detection (PSI — Population Stability Index)
  • Precision/Recall tracking on confirmed fraud labels (from fraud ops team)
  • Alert generation when model performance drops below thresholds
  • Session statistics for the live API (flagged rate, block rate, latency)

PSI is the financial industry's standard metric for detecting feature/score drift.
PSI < 0.10 → No drift | 0.10–0.25 → Moderate drift | > 0.25 → High drift (retrain!)
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
from collections import deque
from threading import Lock
import json

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.logger import get_logger

log = get_logger("monitoring")

# Session Stats (in-memory, reset on API restart) 
class SessionMonitor:
    """
    Tracks live API session statistics.
    Thread-safe (Lock) for concurrent async request handling.
    """

    def __init__(self, window: int = 1000):
        self._lock = Lock()
        self._window = window
        self._scores: deque[float] = deque(maxlen=window)
        self._decisions: deque[str] = deque(maxlen=window)
        self._latencies: deque[float] = deque(maxlen=window)
        self._novel_flags: int = 0
        self._total: int = 0
        self._start_time = datetime.now(timezone.utc)

    def record(self, risk_score: float, decision: str,
               latency_ms: float, is_novel: bool) -> None:
        with self._lock:
            self._scores.append(risk_score)
            self._decisions.append(decision)
            self._latencies.append(latency_ms)
            self._total += 1
            if is_novel:
                self._novel_flags += 1

    def summary(self) -> dict:
        with self._lock:
            if not self._scores:
                return {"status": "no_data"}

            scores = list(self._scores)
            decisions = list(self._decisions)
            latencies = list(self._latencies)
            n = len(scores)

            uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()

            return {
                "total_scored":       self._total,
                "window_size":        n,
                "uptime_seconds":     round(uptime, 1),
                "score_mean":         round(float(np.mean(scores)), 4),
                "score_p95":          round(float(np.percentile(scores, 95)), 4),
                "score_p99":          round(float(np.percentile(scores, 99)), 4),
                "decision_counts": {
                    d: decisions.count(d) for d in ("CLEAR", "REVIEW", "FLAG", "BLOCK")
                },
                "flag_rate_pct":      round(decisions.count("FLAG") / n * 100, 2),
                "block_rate_pct":     round(decisions.count("BLOCK") / n * 100, 2),
                "novel_anomaly_count": self._novel_flags,
                "latency_mean_ms":    round(float(np.mean(latencies)), 2),
                "latency_p99_ms":     round(float(np.percentile(latencies, 99)), 2),
            }

    def check_drift_alert(self, baseline_mean: float = 0.05,
                          threshold: float = 0.15) -> bool:
        """
        Simple mean-shift alert: if the live score mean drifts more than
        `threshold` from the training baseline, trigger a retrain alert.
        """
        with self._lock:
            if len(self._scores) < 100:
                return False
            current_mean = float(np.mean(self._scores))
            drift = abs(current_mean - baseline_mean)
            if drift > threshold:
                log.warning(
                    f"⚠️  SCORE DRIFT ALERT: live mean={current_mean:.4f}  "
                    f"baseline={baseline_mean:.4f}  drift={drift:.4f} > {threshold}"
                )
                return True
            return False


# Population Stability Index 
def compute_psi(
    baseline_scores: np.ndarray,
    current_scores: np.ndarray,
    bins: int = 10,
) -> float:
    """
    Population Stability Index between a baseline score distribution
    (from training/validation) and current live scores.

    PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)

    Interpretation:
      < 0.10 → Stable, no action needed
      0.10–0.25 → Monitor closely, consider retraining
      > 0.25 → Significant drift — retrain immediately
    """
    baseline_scores = np.array(baseline_scores)
    current_scores  = np.array(current_scores)

    # Bin edges from baseline
    _, bin_edges = np.histogram(baseline_scores, bins=bins)
    bin_edges[0]  = -np.inf
    bin_edges[-1] =  np.inf

    baseline_pcts = np.histogram(baseline_scores, bins=bin_edges)[0] / len(baseline_scores)
    current_pcts  = np.histogram(current_scores,  bins=bin_edges)[0] / len(current_scores)

    # Avoid division by zero / log(0)
    baseline_pcts = np.clip(baseline_pcts, 1e-6, None)
    current_pcts  = np.clip(current_pcts,  1e-6, None)

    psi = float(np.sum((current_pcts - baseline_pcts) * np.log(current_pcts / baseline_pcts)))

    status = (
        "STABLE" if psi < 0.10 else
        "MONITOR" if psi < 0.25 else
        "RETRAIN REQUIRED"
    )
    log.info(f"PSI = {psi:.4f}  →  {status}")
    return psi


# Singleton session monitor (used by API)
session_monitor = SessionMonitor()
