"""Evaluation utilities shared across notebooks: fraud-specific metrics,
model-comparison table building, and investigator-capacity ranking."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def fraud_metrics(y_true, y_pred, y_proba) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "fpr": fpr,
        "confusion_matrix": cm,
    }


def metrics_table(results: dict) -> pd.DataFrame:
    """results: {label: fraud_metrics(...) dict} -> tidy comparison DataFrame."""
    rows = []
    for label, m in results.items():
        rows.append(
            {
                "model": label,
                "pr_auc": m["pr_auc"],
                "roc_auc": m["roc_auc"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "fpr": m["fpr"],
            }
        )
    return pd.DataFrame(rows).set_index("model").round(4)


def precision_recall_at_k(y_true, y_proba, k_fracs=(0.005, 0.01, 0.02, 0.05)) -> pd.DataFrame:
    """Precision@K / Recall@K if investigators can only review the top K
    fraction of transactions, ranked by predicted risk score."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    total_fraud = y_true.sum()
    order = np.argsort(-y_proba)

    rows = []
    for k in k_fracs:
        top_n = max(1, int(round(n * k)))
        top_idx = order[:top_n]
        fraud_captured = y_true[top_idx].sum()
        precision_k = fraud_captured / top_n
        recall_k = fraud_captured / total_fraud if total_fraud > 0 else 0.0
        rows.append(
            {
                "review_capacity": f"Top {k * 100:g}%",
                "n_reviewed": top_n,
                "fraud_captured": int(fraud_captured),
                "precision_at_k": round(precision_k, 4),
                "recall_at_k": round(recall_k, 4),
            }
        )
    return pd.DataFrame(rows).set_index("review_capacity")
