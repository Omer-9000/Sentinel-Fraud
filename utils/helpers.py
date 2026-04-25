"""
utils/helpers.py — Shared Utility Functions for SentinelFraud
"""

import hashlib
import json
import time
import uuid
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from utils.logger import get_logger

log = get_logger("helpers")


def generate_transaction_id(prefix: str = "TXN") -> str:
    """Generate a collision-resistant transaction ID."""
    ts = int(time.time() * 1000)
    uid = uuid.uuid4().hex[:8].upper()
    return f"{prefix}-{ts}-{uid}"


def timer(label: str = "") -> Callable:
    """Decorator that logs the execution time of any function."""
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = (time.perf_counter() - t0) * 1000
            log.debug(f"{label or fn.__name__} completed in {elapsed:.2f}ms")
            return result
        return wrapper
    return decorator


def safe_json_dump(obj: Any, path: Path, indent: int = 2) -> None:
    """Serialize an object to JSON, handling numpy types gracefully."""
    class _NumpyEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)):  return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.ndarray):     return o.tolist()
            return super().default(o)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, cls=_NumpyEncoder, indent=indent)
    log.debug(f"Saved JSON to {path}")


def fingerprint_dataframe(df: pd.DataFrame) -> str:
    """
    Compute a stable hash of a DataFrame's content.
    Used for data versioning / cache invalidation.
    """
    h = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
    return h[:12]


def describe_imbalance(y: pd.Series) -> str:
    """Return a human-readable summary of class imbalance."""
    n_fraud = int(y.sum())
    n_total = len(y)
    ratio   = n_fraud / n_total
    return (
        f"{n_fraud:,} fraud / {n_total:,} total "
        f"({ratio*100:.3f}% fraud rate, "
        f"imbalance ratio 1:{int((n_total - n_fraud) / max(n_fraud, 1))})"
    )
