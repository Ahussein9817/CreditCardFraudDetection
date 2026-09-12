"""Evaluation utilities shared across notebooks: fraud-specific metrics
and model-comparison table building."""

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
