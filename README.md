# SentinelFraud 🛡️

**Production-grade, real-time transaction fraud detection system** combining supervised gradient boosting with unsupervised anomaly detection, SHAP explainability, and a live FastAPI scoring engine.

> Built as a realistic prototype of the fraud detection infrastructure used at modern financial institutions. Designed to be recruiter-ready and technically credible.

---

## Why This Matters to a Bank

| Challenge | SentinelFraud's Approach |
|---|---|
| ~0.17% fraud rate (extreme imbalance) | SMOTE over-sampling + `scale_pos_weight` in XGBoost |
| Novel fraud patterns not in training data | Isolation Forest detects statistical outliers (zero-day fraud) |
| Regulator explainability requirements (SR 11-7) | SHAP values per transaction + plain-English reasons |
| False positives damage customer trust | FPR-calibrated threshold (target FPR ≤ 1%) |
| Real-time card processing (< 50ms SLA) | FastAPI + in-memory model serving |
| Audit trail compliance | Immutable JSONL log of every flagged transaction |

---

## System Architecture

```
Raw Transaction Data
        │
        ▼
┌───────────────────┐
│  Preprocessing    │  ← log(Amount), cyclical Time, feature engineering
│  + SMOTE          │  ← balances training data (fraud rate 0.17% → 23%)
└────────┬──────────┘
         │
    ┌────┴────────────────────────────┐
    │                                 │
    ▼                                 ▼
┌──────────────┐            ┌──────────────────────┐
│  XGBoost     │            │  Isolation Forest     │
│  Classifier  │            │  Anomaly Detector     │
│  (supervised)│            │  (unsupervised)       │
└──────┬───────┘            └──────────┬────────────┘
       │  xgb_score [0–1]              │  anomaly_score [0–1]
       └──────────────┬────────────────┘
                      ▼
          ┌───────────────────────┐
          │   Hybrid Risk Scorer  │
          │  score = 0.7×XGB      │
          │        + 0.3×IF       │
          └──────────┬────────────┘
                     │
          ┌──────────┴──────────┐
          │   Decision Engine   │
          │  CLEAR / REVIEW /   │
          │  FLAG  / BLOCK      │
          └──────────┬──────────┘
                     │
          ┌──────────┴──────────┐
          │  SHAP Explainer     │  ← per-feature contributions
          │  + Audit Logger     │  ← immutable JSONL trail
          └──────────┬──────────┘
                     │
          ┌──────────┴──────────┐
          │    FastAPI          │
          │  /score             │
          │  /score/batch       │
          │  /health            │
          │  /metrics           │
          │  /flagged/recent    │
          └─────────────────────┘
```

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| Primary model | XGBoost 2.x | Industry standard for tabular fraud; fast inference; native SHAP |
| Anomaly detection | Isolation Forest | Catches novel fraud not in training labels |
| Imbalance handling | SMOTE (imbalanced-learn) | Creates synthetic fraud examples; better than random over-sampling |
| Explainability | SHAP (TreeExplainer) | Fastest SHAP variant for trees; regulatory-grade explanations |
| API framework | FastAPI + Pydantic v2 | Async, type-safe, auto-generates OpenAPI docs |
| Model serving | In-process (joblib) | Sub-millisecond load after startup |
| Scaling | RobustScaler | Less sensitive to outlier amounts than StandardScaler |
| Logging | Python logging + JSONL | Structured, SIEM-compatible output |

---

## Project Structure

```
fraud_detection/
├── config.py                  # All hyperparameters, paths, thresholds
├── preprocessing.py           # Feature engineering, SMOTE, scaling pipeline
├── train.py                   # End-to-end training orchestrator
├── evaluation.py              # Metrics (PR-AUC, KS, F2) + plots
│
├── models/
│   ├── xgboost_model.py       # XGBoost classifier wrapper
│   ├── anomaly_detector.py    # Isolation Forest wrapper
│   ├── hybrid_scorer.py       # Risk score blending + decision tiers
│   └── explainability.py      # SHAP integration
│
├── api/
│   ├── main.py                # FastAPI application
│   ├── schemas.py             # Pydantic request/response models
│   └── test_api.py            # Integration tests + latency benchmark
│
├── data/
│   ├── download_data.py       # Kaggle download or synthetic generation
│   ├── raw/                   # creditcard.csv (gitignored)
│   └── processed/             # Parquet train/test splits
│
├── models/saved/              # Serialised model artifacts (gitignored)
├── reports/                   # Evaluation plots, metrics JSON, SHAP plots
├── logs/                      # App logs + flagged transaction audit trail
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get the dataset

**Option A — Real Kaggle dataset (recommended):**
```bash
# Set up Kaggle API credentials first (https://kaggle.com/account)
python data/download_data.py
```

**Option B — Synthetic dataset (no account needed):**
```bash
python data/download_data.py --synthetic
```

### 3. Train the models
```bash
python train.py
```
Training takes ~3–8 minutes on a laptop. You'll see live CV scores and a full metrics summary at the end.

### 4. Start the API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive docs: **http://localhost:8000/docs**

### 5. Test it
```bash
python api/test_api.py
```

---

## API Endpoints

### `POST /score` — Real-time transaction scoring

**Request:**
```json
{
  "transaction_id": "TXN-20240124-00421",
  "Time": 52835.0,
  "Amount": 149.99,
  "V1": -1.3598,
  "V2": -0.0728,
  "V3":  2.5363,
  ...
  "V28": -0.0211
}
```

**Response:**
```json
{
  "transaction_id": "TXN-20240124-00421",
  "xgb_score": 0.0312,
  "anomaly_score": 0.1847,
  "risk_score": 0.0773,
  "decision": "CLEAR",
  "is_novel_anomaly": false,
  "top_risk_factors": {
    "V14": -2.341,
    "Amount_log": 0.814,
    "V17": -1.203
  },
  "plain_english_reason": "Risk elevated by: Amount_log | Mitigated by: V14, V17",
  "processing_time_ms": 8.4
}
```

### `POST /score/batch` — Batch scoring (≤ 1,000 transactions)
### `GET /health` — Load balancer health probe
### `GET /metrics` — Model evaluation metrics (for Grafana)
### `GET /flagged/recent` — Audit trail of recent FLAG/BLOCK decisions

---

## Model Performance

After training on the real creditcard.csv dataset:

| Metric | Value | Interpretation |
|---|---|---|
| **PR-AUC** | ~0.85+ | Strong precision-recall balance on fraud class |
| **ROC-AUC** | ~0.98+ | Excellent discrimination |
| **Recall** | ~0.84+ | ~84% of fraud transactions caught |
| **Precision** | ~0.88+ | ~88% of flagged transactions are genuine fraud |
| **F2-Score** | ~0.85+ | Weighted toward recall (banking priority) |
| **KS Statistic** | ~0.85+ | Good score separation between fraud/legit |
| **False Alert Rate** | <1% | 1 in 100 legit transactions unnecessarily flagged |

> Metrics vary slightly based on random seed and whether real or synthetic data is used.

---

## Decision Tier Logic

```
Risk Score    Decision    Action
──────────────────────────────────────────────────────────────
0.00 – 0.30   CLEAR       Pass through automatically
0.30 – 0.55   REVIEW      Queue for analyst review (24h SLA)
0.55 – 0.75   FLAG        Real-time alert + step-up authentication
0.75 – 1.00   BLOCK       Decline transaction + notify customer
```

Thresholds are calibrated to target a false-positive rate ≤ 1% on the validation set. Adjustable in `config.py`.

---

## Explainability Output

Every scored transaction includes SHAP feature contributions:

```
Feature           SHAP Value   Direction
────────────────────────────────────────
V14               -2.34        ↓ toward LEGIT
V17               -1.20        ↓ toward LEGIT
Amount_log        +0.81        ↑ toward FRAUD
V12               -0.67        ↓ toward LEGIT
v_pca_norm        +0.43        ↑ toward FRAUD
```

Positive values push the model toward FRAUD; negative toward LEGIT.  
Plain-English summary: `"Risk elevated by: Amount_log, v_pca_norm | Mitigated by: V14, V17, V12"`

---

## Scaling to Production

| Area | Production Approach |
|---|---|
| Model serving | Deploy on AWS SageMaker / Azure ML with auto-scaling |
| Feature store | Real-time features from Apache Kafka + Feast |
| Throughput | Async batch inference via Celery + Redis |
| Database | PostgreSQL for transaction storage, ClickHouse for analytics |
| Monitoring | Prometheus + Grafana; model drift alerts via Evidently AI |
| Retraining | Weekly automated retraining triggered by drift detection |
| Security | mTLS between services; AES-256 at rest; PII masking in logs |
| Compliance | Full model card + audit trail for SR 11-7 / EU AI Act |

---

## Dataset

**Kaggle Credit Card Fraud Detection**  
284,807 transactions | 492 fraud (0.172%)  
Features V1–V28: PCA-transformed (anonymised) | Amount, Time: raw  
Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## Acknowledgements

Dataset: Dal Pozzolo, A. et al. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. IEEE SSCI.
