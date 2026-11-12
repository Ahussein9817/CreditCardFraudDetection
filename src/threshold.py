"""Threshold sweep and constrained threshold selection.

Extracted verbatim (logic unchanged) from notebooks/03_threshold_analysis.ipynb
so it can be unit tested. The notebook still contains its own copy and its
saved outputs are the source of the README numbers.
"""

import numpy as np
import pandas as pd

from evaluate import fraud_metrics


def threshold_sweep(y_true, y_proba, thresholds) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    rows = []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        m = fraud_metrics(y_true, pred, y_proba)
        n_alerts = int(pred.sum())
        rows.append(
            {
                "threshold": t,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "fpr": m["fpr"],
                "n_alerts": n_alerts,
                "pct_flagged": 100 * n_alerts / n,
            }
        )
    return pd.DataFrame(rows)


def best_threshold_under_fpr_cap(sweep_df: pd.DataFrame, fpr_cap: float):
    """Row with the highest recall among thresholds with fpr <= fpr_cap,
    or None if no threshold satisfies the cap. Ties resolve to the first
    (lowest) threshold in sweep order."""
    feasible = sweep_df[sweep_df["fpr"] <= fpr_cap]
    if feasible.empty:
        return None
    return feasible.loc[feasible["recall"].idxmax()]
