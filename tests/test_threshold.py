import numpy as np
import pytest

import threshold as th


@pytest.fixture
def scored():
    # 4 frauds, 96 legit; frauds score high, a few legit score mid.
    y = np.array([1] * 4 + [0] * 96)
    p = np.array([0.95, 0.9, 0.6, 0.3] + [0.5] * 2 + [0.1] * 94)
    return y, p


def test_sweep_columns_and_monotonic_alerts(scored):
    y, p = scored
    df = th.threshold_sweep(y, p, [0.05, 0.2, 0.55, 0.99])
    assert list(df["threshold"]) == [0.05, 0.2, 0.55, 0.99]
    assert df["n_alerts"].is_monotonic_decreasing
    assert df.loc[df["threshold"] == 0.99, "n_alerts"].item() == 0


def test_sweep_known_values(scored):
    y, p = scored
    row = th.threshold_sweep(y, p, [0.55]).iloc[0]
    # flagged: 0.95, 0.9, 0.6 (3 frauds) -> recall 3/4, no FP, precision 1
    assert row["n_alerts"] == 3
    assert row["recall"] == pytest.approx(0.75)
    assert row["precision"] == pytest.approx(1.0)
    assert row["fpr"] == 0.0
    assert row["pct_flagged"] == pytest.approx(3.0)


def test_best_threshold_picks_max_recall_under_cap(scored):
    y, p = scored
    df = th.threshold_sweep(y, p, [0.05, 0.2, 0.35, 0.55])
    best = th.best_threshold_under_fpr_cap(df, fpr_cap=0.025)
    # 0.2 -> FP=2 (the 0.5 legit) fpr=2/96=0.0208 recall 4/4? 0.3>=0.2 yes -> recall 1.0
    assert best["threshold"] == 0.2
    assert best["recall"] == pytest.approx(1.0)
    assert best["fpr"] <= 0.025


def test_best_threshold_none_when_cap_infeasible(scored):
    y, p = scored
    df = th.threshold_sweep(y, p, [0.05])  # flags everything, fpr = 1
    assert th.best_threshold_under_fpr_cap(df, fpr_cap=0.005) is None


def test_best_threshold_zero_cap_allows_zero_fpr_only(scored):
    y, p = scored
    df = th.threshold_sweep(y, p, [0.2, 0.55])
    best = th.best_threshold_under_fpr_cap(df, fpr_cap=0.0)
    assert best["threshold"] == 0.55
