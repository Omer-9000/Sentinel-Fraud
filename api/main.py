"""
api/main.py — FastAPI Application for SentinelFraud

Production-ready API with:
  • Real-time single-transaction scoring (< 10ms target latency)
  • Batch scoring endpoint (up to 1,000 transactions)
  • SHAP-powered explainability per decision
  • /health endpoint for load-balancer probes
  • /metrics endpoint for monitoring dashboards (Grafana, Datadog)
  • Automatic OpenAPI docs at /docs (Swagger UI)
  • Structured request/response logging
  • Flagged transaction audit trail (JSONL)

Start the API:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    # or simply:
    python api/main.py
"""

import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.schemas import (
    BatchFraudScoreResponse,
    BatchTransactionRequest,
    FraudScoreResponse,
    HealthResponse,
    ModelMetricsResponse,
    TransactionRequest,
)
from config import (
    API_HOST,
    API_PORT,
    ISOLATION_FOREST_PATH,
    REPORTS_DIR,
    XGBOOST_PATH,
)
from preprocessing import preprocess_single
from utils.logger import get_logger, log_flagged_transaction

log = get_logger("api")

# Global Model Registry
# Models are loaded once at startup and reused across all requests.
# In production this would be managed by a model registry (MLflow, SageMaker).
_scorer = None
_explainer = None
_startup_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup; clean up at shutdown."""
    global _scorer, _explainer
    log.info("SentinelFraud API starting — loading models …")

    if not XGBOOST_PATH.exists() or not ISOLATION_FOREST_PATH.exists():
        log.warning(
            "Model files not found! Run 'python train.py' first. "
            "API will start in DEGRADED mode."
        )
    else:
        try:
            from models.hybrid_scorer import HybridFraudScorer
            _scorer = HybridFraudScorer().load_models()

            from models.explainability import FraudExplainer
            _explainer = FraudExplainer(_scorer.xgb)

            log.info("All models loaded. API is HEALTHY.")
        except Exception as e:
            log.error(f"Model loading failed: {e}")

    yield  # API runs here

    log.info("SentinelFraud API shutting down.")


# Application
app = FastAPI(
    title="SentinelFraud API",
    description=(
        "Real-time fraud detection API combining XGBoost supervised classification "
        "with Isolation Forest anomaly detection. Provides risk scores, tiered "
        "decisions, and SHAP-powered explainability for every transaction."
    ),
    version="1.0.0",
    contact={
        "name":  "SentinelFraud Team",
        "email": "fraud-ml@sentinel.bank",
    },
    license_info={"name": "Proprietary"},
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # restrict to internal origins in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware: Request Logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    req_id = uuid.uuid4().hex[:8]
    log.debug(f"[{req_id}] {request.method} {request.url.path}")
    start = time.time()
    response = await call_next(request)
    ms = (time.time() - start) * 1000
    log.debug(f"[{req_id}] → {response.status_code}  ({ms:.1f}ms)")
    return response


# Helper
def _check_models_loaded():
    if _scorer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Models not loaded. Run 'python train.py' to train the models first, "
                "then restart the API."
            ),
        )


def _score_transaction(txn: TransactionRequest) -> FraudScoreResponse:
    """Core scoring logic shared by single and batch endpoints."""
    t0 = time.perf_counter()

    # Preprocess to model-ready features
    raw = txn.model_dump()
    X = preprocess_single(raw)

    # Hybrid score
    result = _scorer.score_single(X, transaction_id=txn.transaction_id)

    # SHAP explanation
    top_factors = {}
    plain_reason = ""
    if _explainer is not None:
        try:
            top_factors = _scorer.xgb.feature_names and _explainer.explain_single(X, top_n=8) or {}
            plain_reason = _explainer.explain_in_plain_english(X)
        except Exception as e:
            log.warning(f"SHAP explanation failed for {txn.transaction_id}: {e}")

    latency_ms = (time.perf_counter() - t0) * 1000

    # Audit log for flagged / blocked transactions
    if result.decision in ("FLAG", "BLOCK"):
        log_flagged_transaction({
            "transaction_id": result.transaction_id,
            "decision":       result.decision,
            "risk_score":     result.risk_score,
            "xgb_score":      result.xgb_score,
            "anomaly_score":  result.anomaly_score,
            "is_novel":       result.is_novel_anomaly,
            "amount":         raw.get("Amount"),
            "plain_reason":   plain_reason,
        })
        log.warning(
            f"[{result.decision}] {result.transaction_id}  "
            f"risk={result.risk_score:.3f}  "
            f"novel={'YES' if result.is_novel_anomaly else 'NO'}"
        )

    return FraudScoreResponse(
        transaction_id=result.transaction_id,
        xgb_score=result.xgb_score,
        anomaly_score=result.anomaly_score,
        risk_score=result.risk_score,
        decision=result.decision,
        is_novel_anomaly=result.is_novel_anomaly,
        top_risk_factors=top_factors,
        plain_english_reason=plain_reason,
        processing_time_ms=round(latency_ms, 2),
    )


# Endpoints

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check for load balancers and monitoring systems",
)
async def health_check():
    return HealthResponse(
        status="healthy" if _scorer is not None else "degraded",
        models_loaded=_scorer is not None,
        version="1.0.0",
        uptime_seconds=round(time.time() - _startup_time, 1),
    )


@app.get(
    "/metrics",
    response_model=ModelMetricsResponse,
    tags=["Monitoring"],
    summary="Latest model evaluation metrics from training run",
)
async def get_model_metrics():
    """
    Returns evaluation metrics from the most recent training run.
    Exposed for Grafana / Prometheus scraping in production.
    """
    import json
    metrics_path = REPORTS_DIR / "evaluation_metrics.json"
    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No evaluation metrics found. Run 'python train.py' first.",
        )
    with open(metrics_path) as f:
        data = json.load(f)

    import os
    from datetime import datetime
    mtime = os.path.getmtime(metrics_path)
    evaluated_at = datetime.fromtimestamp(mtime).isoformat()

    return ModelMetricsResponse(
        pr_auc=data.get("pr_auc"),
        roc_auc=data.get("roc_auc"),
        recall=data.get("recall"),
        precision=data.get("precision"),
        f1_score=data.get("f1_score"),
        f2_score=data.get("f2_score"),
        ks_statistic=data.get("ks_statistic"),
        threshold=data.get("threshold"),
        evaluated_at=evaluated_at,
    )


@app.post(
    "/score",
    response_model=FraudScoreResponse,
    tags=["Fraud Detection"],
    summary="Score a single transaction in real time",
    status_code=status.HTTP_200_OK,
)
async def score_transaction(txn: TransactionRequest):
    """
    Submit a single transaction and receive an immediate fraud risk score.

    **Decision tiers:**
    - `CLEAR` (<0.30): Transaction passes through automatically.
    - `REVIEW` (0.30–0.55): Queued for analyst review within 24 hours.
    - `FLAG` (0.55–0.75): Real-time alert triggered; step-up authentication requested.
    - `BLOCK` (>0.75): Transaction declined; customer notified.

    Typical latency: **< 15ms** (P99).
    """
    _check_models_loaded()
    try:
        return _score_transaction(txn)
    except Exception as e:
        log.error(f"Scoring error for {txn.transaction_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


@app.post(
    "/score/batch",
    response_model=BatchFraudScoreResponse,
    tags=["Fraud Detection"],
    summary="Score a batch of up to 1,000 transactions",
    status_code=status.HTTP_200_OK,
)
async def score_batch(batch: BatchTransactionRequest):
    """
    Batch scoring endpoint for end-of-day reconciliation, bulk risk assessment,
    or replaying historical transactions through the updated model.

    Returns aggregated counts plus individual results for each transaction.
    """
    _check_models_loaded()
    t0 = time.perf_counter()
    results = []

    for txn in batch.transactions:
        try:
            results.append(_score_transaction(txn))
        except Exception as e:
            log.error(f"Batch scoring error for {txn.transaction_id}: {e}")
            # Don't abort entire batch on single failure — return partial results
            results.append(
                FraudScoreResponse(
                    transaction_id=txn.transaction_id,
                    xgb_score=0.0,
                    anomaly_score=0.0,
                    risk_score=0.0,
                    decision="CLEAR",
                    is_novel_anomaly=False,
                    top_risk_factors={},
                    plain_english_reason="Scoring error — defaulting to CLEAR",
                    processing_time_ms=0.0,
                )
            )

    total_ms = (time.perf_counter() - t0) * 1000
    flagged = sum(1 for r in results if r.decision == "FLAG")
    blocked = sum(1 for r in results if r.decision == "BLOCK")

    log.info(
        f"Batch scored {len(results)} transactions in {total_ms:.1f}ms — "
        f"FLAG: {flagged}  BLOCK: {blocked}"
    )

    return BatchFraudScoreResponse(
        total=len(results),
        flagged=flagged,
        blocked=blocked,
        results=results,
        processing_time_ms=round(total_ms, 2),
    )


@app.get(
    "/flagged/recent",
    tags=["Monitoring"],
    summary="Retrieve the most recently flagged/blocked transactions",
)
async def get_recent_flagged(limit: int = 20):
    """
    Returns the last N flagged or blocked transactions from the audit log.
    In production this would query a database (e.g., PostgreSQL, BigQuery).
    """
    from config import FLAGGED_LOG
    if not FLAGGED_LOG.exists():
        return {"flagged_transactions": [], "total": 0}

    import json
    lines = FLAGGED_LOG.read_text(encoding="utf-8").strip().splitlines()
    recent = [json.loads(l) for l in lines[-limit:]][::-1]  # newest first
    return {"flagged_transactions": recent, "total": len(lines)}


# Entry Point
if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info",
    )
