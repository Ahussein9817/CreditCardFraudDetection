import numpy as np
import pytest

import bootstrap as bs


def _data(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    good = np.clip(rng.random(n) * 0.8 + y * 0.3, 0, 1)  # imperfect separation (AUC < 1)
    weak = rng.random(n)
    return y, good, weak


def test_metrics_at_known_values():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    m = bs.metrics_at(y, p, 0.5, k_fracs=(0.2,))
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5)
    assert m["fpr"] == pytest.approx(1 / 8)
    assert m["precision_at_20pct"] == pytest.approx(0.5)  # top 2: one fraud
    assert m["recall_at_20pct"] == pytest.approx(0.5)


def test_metrics_single_class_gives_nan_aucs_not_crash():
    m = bs.metrics_at(np.zeros(5, dtype=int), np.linspace(0, 1, 5), 0.5)
    assert np.isnan(m["pr_auc"]) and np.isnan(m["roc_auc"]) and np.isnan(m["recall"])


def test_bootstrap_point_inside_ci_and_deterministic():
    y, p, _ = _data()
    a = bs.bootstrap_metrics(y, p, 0.5, n_boot=100, seed=3)
    b = bs.bootstrap_metrics(y, p, 0.5, n_boot=100, seed=3)
    assert a == b
    for name in ("pr_auc", "roc_auc", "recall"):
        lo, hi = a["metrics"][name]["ci"]
        assert lo <= a["metrics"][name]["point"] <= hi


def test_bootstrap_more_data_gives_narrower_ci():
    ys, ps, _ = _data(n=1000)
    yl, pl, _ = _data(n=8000)
    w = lambda r: r["metrics"]["roc_auc"]["ci"][1] - r["metrics"]["roc_auc"]["ci"][0]  # noqa: E731
    assert w(bs.bootstrap_metrics(yl, pl, 0.5, 150, 1)) < w(bs.bootstrap_metrics(ys, ps, 0.5, 150, 1))


def test_bootstrap_counts_skipped_single_class_resamples():
    y = np.zeros(20, dtype=int)
    y[0] = 1
    r = bs.bootstrap_metrics(y, np.random.default_rng(0).random(20), 0.5, n_boot=100, seed=0)
    assert 0 < r["n_skipped"] < 100


def test_paired_diff_identical_scorers_is_exactly_zero():
    y, p, _ = _data()
    r = bs.paired_bootstrap_diff(y, p, p.copy(), n_boot=50, seed=1)
    assert r["diff_point"] == 0 and r["ci"] == (0.0, 0.0)


def test_paired_diff_good_vs_random_is_clearly_positive():
    y, good, weak = _data()
    r = bs.paired_bootstrap_diff(y, good, weak, metric="roc_auc", n_boot=100, seed=1)
    assert r["ci"][0] > 0.05 and r["pct_positive"] == 1.0
