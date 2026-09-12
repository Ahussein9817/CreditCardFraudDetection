"""Batch-based monitoring and drift detection for the deployed model.

Simulates production by replaying held-out data in consecutive calendar
batches and tracking fraud performance, operational load, and model/data
drift batch over batch. Deliberately lightweight (histogram-based PSI,
a KS test) rather than a full drift-detection platform.
"""

import numpy as np
import pandas as pd
from scipy import stats

from evaluate import fraud_metrics

# Fixed risk-score tiers for PSI, rather than quantile bins: this Random
# Forest's scores are heavily zero-inflated (~94% of validation scores are
# exactly 0.0), so quantile edges up to the 90th percentile would all
# collapse to 0 and PSI would look like 0 drift by construction.
RISK_SCORE_BINS = [0.0, 1e-6, 0.01, 0.05, 0.12, 0.3, 0.5, 0.8, 1.0]


def make_time_batches(df: pd.DataFrame, time_col: str = "trans_date_trans_time", freq: str = "W"):
    """Splits df into consecutive calendar-period batches (default weekly),
    in chronological order."""
    period = df[time_col].dt.to_period(freq)
    return [(str(p), g) for p, g in df.groupby(period)]


def population_stability_index(expected, actual, bins=10) -> float:
    """PSI between two 1-D numeric distributions. PSI < 0.1 ~ no material
    shift, 0.1-0.25 ~ moderate, > 0.25 ~ significant shift.

    `bins` is either an int (quantile bins computed from `expected` - fine
    for a roughly-continuous feature like amount) or explicit bin edges
    (needed for a zero-inflated distribution like this model's risk score,
    where ~94% of scores are exactly 0.0 and quantile bins up to the 90th
    percentile would all collapse to the same edge)."""
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    if np.ndim(bins) == 0:
        edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    else:
        edges = np.asarray(bins, dtype=float)
    edges[0], edges[-1] = -np.inf, np.inf
    exp_counts, _ = np.histogram(expected, bins=edges)
    act_counts, _ = np.histogram(actual, bins=edges)
    exp_pct = np.clip(exp_counts / len(expected), 1e-6, None)
    act_pct = np.clip(act_counts / len(actual), 1e-6, None)
    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


def ks_drift(expected, actual) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test: (statistic, p-value). A small
    p-value means the batch's distribution differs from the reference."""
    result = stats.ks_2samp(expected, actual)
    return float(result.statistic), float(result.pvalue)


def batch_metrics(
    y_true,
    y_proba,
    threshold: float,
    reference_risk_scores=None,
    reference_amt=None,
    batch_amt=None,
) -> dict:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    pred = (y_proba >= threshold).astype(int)
    m = fraud_metrics(y_true, pred, y_proba)
    n = len(y_true)
    n_alerts = int(pred.sum())

    result = {
        "n_transactions": n,
        "n_alerts": n_alerts,
        "pct_flagged": 100 * n_alerts / n,
        "observed_fraud_rate_pct": 100 * float(y_true.mean()),
        "predicted_fraud_rate_pct": 100 * float(pred.mean()),
        "avg_risk_score": float(y_proba.mean()),
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "pr_auc": m["pr_auc"],
        "fpr": m["fpr"],
    }
    if reference_risk_scores is not None:
        result["risk_score_psi"] = population_stability_index(
            reference_risk_scores, y_proba, bins=RISK_SCORE_BINS
        )
    if reference_amt is not None and batch_amt is not None:
        ks_stat, ks_p = ks_drift(reference_amt, batch_amt)
        result["amt_ks_stat"] = ks_stat
        result["amt_ks_pvalue"] = ks_p
    return result


def monitor(
    df: pd.DataFrame,
    y_true_col: str,
    proba_col: str,
    amt_col: str,
    threshold: float,
    reference_risk_scores,
    reference_amt,
    time_col: str = "trans_date_trans_time",
    freq: str = "W",
) -> pd.DataFrame:
    """Runs batch_metrics over consecutive time batches of df. `reference_*`
    should come from the validation set used to tune the model/threshold -
    drift is measured relative to what the model was tuned on, not within
    the batches themselves."""
    rows = []
    for period, batch in make_time_batches(df, time_col=time_col, freq=freq):
        m = batch_metrics(
            batch[y_true_col].values,
            batch[proba_col].values,
            threshold,
            reference_risk_scores=reference_risk_scores,
            reference_amt=reference_amt,
            batch_amt=batch[amt_col].values,
        )
        m["batch"] = period
        rows.append(m)
    return pd.DataFrame(rows).set_index("batch")
