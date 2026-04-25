"""
api/test_api.py - Integration Tests & Sample Requests for SentinelFraud API

Run these tests AFTER starting the API:
    uvicorn api.main:app --port 8000

Then in a new terminal:
    python api/test_api.py

Tests cover:
  * Health check
  * Single legitimate transaction
  * Single high-risk fraud transaction
  * Batch scoring with mixed transactions
  * Edge cases (zero amount, large amount)
  * Model metrics retrieval
  * Recent flagged transactions
"""

import json
import sys
import time
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000"

# Sample Transactions
# These V-feature values are based on statistical profiles from the real dataset.
# Legitimate transaction: V-features near zero (average PCA components)
LEGIT_TRANSACTION = {
    "transaction_id": "TXN-TEST-LEGIT-001",
    "Time": 86400.0,        # 24 hours in
    "Amount": 45.50,        # typical grocery purchase
    "V1":  0.312,  "V2":  0.178,  "V3": -0.245, "V4":  0.456,
    "V5":  0.123,  "V6": -0.089,  "V7":  0.201, "V8":  0.034,
    "V9": -0.112,  "V10": 0.067,  "V11": 0.289, "V12": 0.156,
    "V13": 0.078,  "V14": 0.234,  "V15": 0.089, "V16": 0.123,
    "V17": 0.045,  "V18": 0.067,  "V19": 0.012, "V20": 0.034,
    "V21": 0.011,  "V22": 0.023,  "V23": 0.005, "V24": 0.017,
    "V25": 0.009,  "V26": 0.014,  "V27": 0.003, "V28": 0.006,
}

# High-risk fraud transaction: extreme V-features typical of real fraud samples
HIGH_RISK_TRANSACTION = {
    "transaction_id": "TXN-TEST-FRAUD-001",
    "Time": 406.0,          # very early in day (3am = suspicious)
    "Amount": 1.00,         # small test charge (card-testing attack pattern)
    "V1": -3.043541,  "V2":  1.190587,  "V3": -3.049668, "V4": -1.221699,
    "V5": -0.293424,  "V6": -0.303556,  "V7": -0.456738, "V8": -0.183854,
    "V9":  0.107074,  "V10":-0.657649,  "V11":-1.340764, "V12":-2.671888,
    "V13": 0.640681,  "V14":-2.738281,  "V15": 0.098741, "V16":-0.856892,
    "V17":-2.614267,  "V18": 0.038431,  "V19":-0.019529, "V20":-0.087403,
    "V21": 0.124101,  "V22":-0.388047,  "V23": 0.017502, "V24": 0.085928,
    "V25":-0.052673,  "V26":-0.097985,  "V27":-0.002972, "V28":-0.025488,
}

LARGE_AMOUNT_TRANSACTION = {
    **LEGIT_TRANSACTION,
    "transaction_id": "TXN-TEST-LARGE-001",
    "Amount": 9500.00,      # large purchase - should elevate risk score
}


# Test Runner
def print_response(label: str, resp: requests.Response, indent: bool = True) -> None:
    print(f"\n{'-'*60}")
    print(f"  TEST: {label}")
    print(f"  Status: {resp.status_code}")
    body = resp.json()
    if indent:
        print(json.dumps(body, indent=2))
    else:
        print(body)
    print(f"{'-'*60}")
    return body


def test_health():
    print("\n" + "="*60)
    print("  SentinelFraud API Integration Tests")
    print("="*60)

    resp = requests.get(f"{BASE_URL}/health")
    body = print_response("GET /health", resp)
    assert resp.status_code == 200, "Health check failed!"
    assert body["status"] in ("healthy", "degraded")
    print("  PASSED")
    return body["models_loaded"]


def test_legit_transaction():
    resp = requests.post(f"{BASE_URL}/score", json=LEGIT_TRANSACTION)
    body = print_response("POST /score - Legitimate transaction ($45.50)", resp)
    assert resp.status_code == 200
    assert body["decision"] in ("CLEAR", "REVIEW")
    assert body["risk_score"] < 0.60, f"Expected low risk, got {body['risk_score']}"
    print("  PASSED - Low risk as expected")
    return body


def test_high_risk_transaction():
    resp = requests.post(f"{BASE_URL}/score", json=HIGH_RISK_TRANSACTION)
    body = print_response("POST /score - High-risk fraud transaction ($1.00)", resp)
    assert resp.status_code == 200
    # A well-trained model should flag this - accept REVIEW/FLAG/BLOCK
    assert body["decision"] != "CLEAR", \
        f"Expected REVIEW/FLAG/BLOCK, got CLEAR (risk={body['risk_score']})"
    print(f"  PASSED - Decision: {body['decision']} (risk={body['risk_score']:.3f})")
    return body


def test_large_amount():
    resp = requests.post(f"{BASE_URL}/score", json=LARGE_AMOUNT_TRANSACTION)
    body = print_response("POST /score - Large amount transaction ($9,500)", resp)
    assert resp.status_code == 200
    print(f"  PASSED - Decision: {body['decision']} (risk={body['risk_score']:.3f})")
    return body


def test_batch_scoring():
    batch_payload = {
        "transactions": [
            LEGIT_TRANSACTION,
            HIGH_RISK_TRANSACTION,
            LARGE_AMOUNT_TRANSACTION,
        ]
    }
    resp = requests.post(f"{BASE_URL}/score/batch", json=batch_payload)
    body = print_response("POST /score/batch - 3 mixed transactions", resp)
    assert resp.status_code == 200
    assert body["total"] == 3
    print(
        f"  PASSED - {body['total']} scored, "
        f"{body['flagged']} flagged, {body['blocked']} blocked "
        f"in {body['processing_time_ms']:.1f}ms"
    )
    return body


def test_model_metrics():
    resp = requests.get(f"{BASE_URL}/metrics")
    if resp.status_code == 404:
        print("\n  [WARN] /metrics - No metrics found (run train.py first)")
        return
    body = print_response("GET /metrics - Model evaluation metrics", resp)
    assert resp.status_code == 200
    print(
        f"  PASSED - PR-AUC: {body.get('pr_auc')}, "
        f"ROC-AUC: {body.get('roc_auc')}"
    )


def test_flagged_recent():
    resp = requests.get(f"{BASE_URL}/flagged/recent?limit=5")
    body = print_response("GET /flagged/recent - Recent flagged transactions", resp)
    assert resp.status_code == 200
    print(f"  PASSED - {body['total']} total flagged transactions in audit log")


def test_latency_benchmark(n: int = 100):
    """Quick latency benchmark - target P99 < 50ms for real-time scoring."""
    print(f"\n{'-'*60}")
    print(f"  BENCHMARK: {n} sequential /score requests")
    latencies = []
    for _ in range(n):
        t0 = time.perf_counter()
        requests.post(f"{BASE_URL}/score", json=LEGIT_TRANSACTION)
        latencies.append((time.perf_counter() - t0) * 1000)

    import statistics
    print(f"  Mean:   {statistics.mean(latencies):.1f}ms")
    print(f"  Median: {statistics.median(latencies):.1f}ms")
    print(f"  P95:    {sorted(latencies)[int(n*0.95)]:.1f}ms")
    print(f"  P99:    {sorted(latencies)[int(n*0.99)]:.1f}ms")
    print(f"  Max:    {max(latencies):.1f}ms")
    print(f"{'-'*60}")


if __name__ == "__main__":
    print(f"\nConnecting to {BASE_URL} ...")

    models_loaded = test_health()

    if not models_loaded:
        print(
            "\n[WARN] Models not loaded. Run 'python train.py' first, "
            "then restart the API."
        )
        print("   Skipping scoring tests.\n")
        sys.exit(0)

    test_legit_transaction()
    test_high_risk_transaction()
    test_large_amount()
    test_batch_scoring()
    test_model_metrics()
    test_flagged_recent()
    test_latency_benchmark(n=50)

    print("\n" + "="*60)
    print("  All tests completed successfully! OK")
    print("="*60 + "\n")
