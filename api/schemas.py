"""
api/schemas.py — Pydantic Request/Response Models for SentinelFraud API

Using Pydantic v2 for automatic validation, type coercion, and OpenAPI docs.
These schemas serve as the contract between the API and any consuming system
(mobile app, core banking system, card processor, etc.)
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


# Inbound: Transaction to Score
class TransactionRequest(BaseModel):
    """
    Represents a single card transaction submitted for fraud scoring.
    V1–V28 are PCA-transformed features (anonymised in production deployments).
    In a real bank integration, these would be computed upstream by the
    feature store from raw transaction metadata.
    """
    transaction_id: str = Field(
        default="auto",
        description="Unique identifier for this transaction. Auto-generated if omitted.",
        examples=["TXN-20240124-00421"],
    )
    Time: float = Field(
        ...,
        ge=0,
        description="Seconds elapsed since the first transaction in the dataset window.",
        examples=[52835.0],
    )
    Amount: float = Field(
        ...,
        ge=0,
        le=100_000,
        description="Transaction amount in USD (or local currency equivalent).",
        examples=[149.99],
    )

    # PCA-transformed anonymised features V1–V28
    V1:  float = Field(..., examples=[-1.359807134])
    V2:  float = Field(..., examples=[-0.072781173])
    V3:  float = Field(..., examples=[2.536346738])
    V4:  float = Field(..., examples=[1.378155224])
    V5:  float = Field(..., examples=[-0.338320769])
    V6:  float = Field(..., examples=[0.462387778])
    V7:  float = Field(..., examples=[0.239598554])
    V8:  float = Field(..., examples=[0.098697901])
    V9:  float = Field(..., examples=[-0.379492080])
    V10: float = Field(..., examples=[-0.387143526])
    V11: float = Field(..., examples=[-0.054951938])
    V12: float = Field(..., examples=[-0.226487264])
    V13: float = Field(..., examples=[0.178228220])
    V14: float = Field(..., examples=[0.507756870])
    V15: float = Field(..., examples=[-0.287923507])
    V16: float = Field(..., examples=[-0.631418118])
    V17: float = Field(..., examples=[0.007206200])
    V18: float = Field(..., examples=[0.044519692])
    V19: float = Field(..., examples=[0.166928798])
    V20: float = Field(..., examples=[0.125894532])
    V21: float = Field(..., examples=[-0.008983099])
    V22: float = Field(..., examples=[0.014724169])
    V23: float = Field(..., examples=[-0.032609400])
    V24: float = Field(..., examples=[-0.038212200])
    V25: float = Field(..., examples=[0.013648600])
    V26: float = Field(..., examples=[-0.021053100])
    V27: float = Field(..., examples=[0.000000000])
    V28: float = Field(..., examples=[-0.021053100])

    @field_validator("transaction_id", mode="before")
    @classmethod
    def set_auto_id(cls, v):
        if v == "auto" or not v:
            import uuid
            return f"TXN-{uuid.uuid4().hex[:12].upper()}"
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "transaction_id": "TXN-20240124-00421",
                "Time": 52835.0,
                "Amount": 149.99,
                **{f"V{i}": round((-1) ** i * 0.5 * i, 4) for i in range(1, 29)},
            }
        }
    }


# Outbound: Fraud Score Response
class FraudScoreResponse(BaseModel):
    """
    Structured fraud scoring result returned for each transaction.
    Designed to be consumed by card processing systems, analyst dashboards,
    and mobile banking apps.
    """
    transaction_id: str
    xgb_score: float = Field(
        description="XGBoost fraud probability [0–1]. Supervised signal.",
    )
    anomaly_score: float = Field(
        description="Isolation Forest anomaly score [0–1]. Unsupervised signal.",
    )
    risk_score: float = Field(
        description="Blended hybrid risk score [0–1]. Primary decision signal.",
    )
    decision: Literal["CLEAR", "REVIEW", "FLAG", "BLOCK"] = Field(
        description=(
            "CLEAR=pass through | REVIEW=analyst queue | "
            "FLAG=alert+step-up auth | BLOCK=decline transaction"
        ),
    )
    is_novel_anomaly: bool = Field(
        description=(
            "True if Isolation Forest flags this as an outlier but XGBoost "
            "gives low probability — may indicate a novel, unseen fraud pattern."
        ),
    )
    top_risk_factors: dict[str, float] = Field(
        default_factory=dict,
        description="Top SHAP feature contributions (positive = toward fraud).",
    )
    plain_english_reason: str = Field(
        default="",
        description="Human-readable explanation for analyst/customer service use.",
    )
    processing_time_ms: float = Field(
        description="End-to-end API latency in milliseconds.",
    )


# Batch Request / Response
class BatchTransactionRequest(BaseModel):
    """Submit up to 1,000 transactions in a single API call."""
    transactions: list[TransactionRequest] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of transactions to score.",
    )


class BatchFraudScoreResponse(BaseModel):
    total: int
    flagged: int
    blocked: int
    results: list[FraudScoreResponse]
    processing_time_ms: float


# Health Check
class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "down"]
    models_loaded: bool
    version: str
    uptime_seconds: float


# Model Metrics
class ModelMetricsResponse(BaseModel):
    pr_auc: Optional[float] = None
    roc_auc: Optional[float] = None
    recall: Optional[float] = None
    precision: Optional[float] = None
    f1_score: Optional[float] = None
    f2_score: Optional[float] = None
    ks_statistic: Optional[float] = None
    threshold: Optional[float] = None
    evaluated_at: Optional[str] = None
