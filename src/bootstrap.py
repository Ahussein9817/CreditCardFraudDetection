"""Bootstrap confidence intervals and paired comparisons for fraud metrics.

Plain row-level (iid) bootstrap. Caveat: transactions from the same cardholder
are correlated, and the committed scored files carry no card id, so a
cluster-by-card bootstrap is not possible; intervals are therefore likely a
little narrower than the truth.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def metrics_at(y, p, threshold, k_fracs=(0.005, 0.01)):
    """Metrics used in the README: PR-AUC, ROC-AUC, precision/recall/FPR at a fixed
    threshold, and Precision@K / Recall@K. NaN where undefined."""
    y = np.asarray(y)
    p = np.asarray(p)
    n_pos = int(y.sum())
    out = {"pr_auc": np.nan, "roc_auc": np.nan}
    if 0 < n_pos < len(y):
        out["pr_auc"] = float(average_precision_score(y, p))
        out["roc_auc"] = float(roc_auc_score(y, p))
    pred = p >= threshold
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    n_neg = len(y) - n_pos
    out["precision"] = tp / pred.sum() if pred.sum() else 0.0
    out["recall"] = tp / n_pos if n_pos else np.nan
    out["fpr"] = fp / n_neg if n_neg else np.nan
    order = np.argsort(-p, kind="stable")
    for k in k_fracs:
        top = max(1, int(round(len(y) * k)))
        hit = int(y[order[:top]].sum())
        out[f"precision_at_{k * 100:g}pct"] = hit / top
        out[f"recall_at_{k * 100:g}pct"] = hit / n_pos if n_pos else np.nan
    return out


def _ci(values, alpha):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2))))


def bootstrap_metrics(y, p, threshold, n_boot=500, seed=0, alpha=0.05):
    """Point estimate and percentile CI for every metric in `metrics_at`.
    Single-class resamples are skipped (and counted)."""
    y = np.asarray(y)
    p = np.asarray(p)
    rng = np.random.default_rng(seed)
    point = metrics_at(y, p, threshold)
    draws = {k: [] for k in point}
    skipped = 0
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() in (0, n):
            skipped += 1
            continue
        m = metrics_at(y[idx], p[idx], threshold)
        for k, v in m.items():
            draws[k].append(v)
    return {
        "n_boot": n_boot, "n_skipped": skipped, "seed": seed, "alpha": alpha,
        "metrics": {k: {"point": point[k], "ci": _ci(draws[k], alpha)} for k in point},
    }


def paired_bootstrap_diff(y, p_a, p_b, metric="pr_auc", threshold=0.5, n_boot=500, seed=0, alpha=0.05):
    """Paired bootstrap of metric(A) - metric(B): the same resampled rows are used
    for both scorers, so the interval reflects their correlation."""
    y = np.asarray(y)
    p_a = np.asarray(p_a)
    p_b = np.asarray(p_b)
    rng = np.random.default_rng(seed)
    n = len(y)
    point = metrics_at(y, p_a, threshold)[metric] - metrics_at(y, p_b, threshold)[metric]
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() in (0, n):
            continue
        diffs.append(metrics_at(y[idx], p_a[idx], threshold)[metric]
                     - metrics_at(y[idx], p_b[idx], threshold)[metric])
    return {"metric": metric, "diff_point": float(point), "ci": _ci(diffs, alpha),
            "pct_positive": float(np.mean(np.array(diffs) > 0)) if diffs else np.nan,
            "n_valid": len(diffs), "n_boot": n_boot, "seed": seed}
