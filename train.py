"""
train.py — End-to-End Training Pipeline for SentinelFraud

Run this script to train both models from scratch:
    python train.py

What this script does:
  1. Download/verify dataset
  2. Run full preprocessing pipeline
  3. Train XGBoost classifier (with cross-validation)
  4. Train Isolation Forest anomaly detector
  5. Evaluate on held-out test set
  6. Generate evaluation plots and SHAP analysis
  7. Save all models and artifacts
"""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    FEATURE_LIST_PATH,
    ISOLATION_FOREST_PATH,
    RAW_DATA_PATH,
    TEST_PATH,
    TRAIN_PATH,
    XGBOOST_PATH,
)
from utils.logger import get_logger

log = get_logger("train")


def main() -> None:
    start = time.time()
    log.info("=" * 60)
    log.info("  SentinelFraud — Training Pipeline Starting")
    log.info("=" * 60)

    # Ensure Dataset Exists 
    if not RAW_DATA_PATH.exists():
        log.info("Dataset not found — generating synthetic dataset …")
        from data.download_data import generate_synthetic_dataset
        generate_synthetic_dataset()

    # Preprocessing
    log.info("\n[Step 2/6] Running preprocessing pipeline …")
    from preprocessing import run_preprocessing_pipeline
    X_train, X_test, y_train, y_test = run_preprocessing_pipeline()

    # Train XGBoost 
    log.info("\n[Step 3/6] Training XGBoost classifier …")
    from models.xgboost_model import FraudXGBoostClassifier
    xgb_clf = FraudXGBoostClassifier()

    # 5-fold cross-validation for robust evaluation
    log.info("  Running 5-fold stratified cross-validation …")
    cv_results = xgb_clf.cross_validate(X_train, y_train, n_splits=5)

    # Full training on entire training set
    xgb_clf.train(X_train, y_train)
    xgb_clf.save(XGBOOST_PATH)

    # Train Isolation Forest
    log.info("\n[Step 4/6] Training Isolation Forest anomaly detector …")
    from models.anomaly_detector import FraudAnomalyDetector
    if_det = FraudAnomalyDetector()
    if_det.train(X_train)
    if_det.save(ISOLATION_FOREST_PATH)

    # Evaluate anomaly detector standalone
    if_metrics = if_det.evaluate(X_test, y_test)

    # Evaluate Hybrid System
    log.info("\n[Step 5/6] Evaluating hybrid scoring system …")
    from models.hybrid_scorer import HybridFraudScorer
    scorer = HybridFraudScorer(xgb_model=xgb_clf, anomaly_detector=if_det)

    # Score the full test set
    score_df = scorer.score_to_dataframe(X_test)
    risk_scores = score_df["risk_score"].values
    xgb_probs   = xgb_clf.predict_fraud_score(X_test)

    from evaluation import generate_full_report
    report = generate_full_report(
        y_true=y_test.values,
        y_proba=risk_scores,
        threshold=xgb_clf.optimal_threshold,
        model_name="SentinelFraud Hybrid (XGBoost + Isolation Forest)",
    )

    # SHAP Explainability
    log.info("\n[Step 6/6] Generating SHAP explanations …")
    try:
        from models.explainability import FraudExplainer
        explainer = FraudExplainer(xgb_clf)

        # Global importance on a 2,000-sample subset (for speed)
        sample_size = min(2000, len(X_test))
        X_sample = X_test.sample(n=sample_size, random_state=42)
        global_imp = explainer.global_feature_importance(X_sample)
        log.info(f"\nTop 10 features by mean |SHAP|:\n{global_imp.head(10).to_string(index=False)}")

        shap_plot = explainer.plot_summary(X_sample)
        log.info(f"SHAP summary plot: {shap_plot}")
    except Exception as e:
        log.warning(f"SHAP analysis skipped: {e}")

    # Summary
    elapsed = time.time() - start
    m = report["metrics"]
    log.info(
        f"\n{'='*60}\n"
        f"  TRAINING COMPLETE  ({elapsed:.1f}s)\n"
        f"{'='*60}\n"
        f"  Hybrid PR-AUC:   {m['pr_auc']:.4f}\n"
        f"  Hybrid ROC-AUC:  {m['roc_auc']:.4f}\n"
        f"  Recall (Fraud):  {m['recall']:.4f}  ({m['fraud_caught_pct']:.1f}% caught)\n"
        f"  False Alerts:    {m['false_alert_pct']:.2f}%\n"
        f"  F2-Score:        {m['f2_score']:.4f}\n"
        f"  KS Statistic:    {m['ks_statistic']:.4f}\n"
        f"\n  CV PR-AUC:  {cv_results['pr_auc_mean']:.4f} ± {cv_results['pr_auc_std']:.4f}\n"
        f"  CV ROC-AUC: {cv_results['roc_auc_mean']:.4f} ± {cv_results['roc_auc_std']:.4f}\n"
        f"\n  IF ROC-AUC: {if_metrics['if_roc_auc']:.4f}\n"
        f"  IF Fraud Recall: {if_metrics['if_fraud_recall']:.4f}\n"
        f"{'='*60}\n"
        f"  Reports saved to: reports/\n"
        f"  Models saved to:  models/saved/\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
