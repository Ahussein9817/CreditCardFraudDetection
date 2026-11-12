import numpy as np
import pandas as pd
import pytest

import monitoring as mon


def test_psi_identical_distributions_is_zero():
    x = np.random.default_rng(0).normal(size=5000)
    assert mon.population_stability_index(x, x) == pytest.approx(0.0, abs=1e-9)


def test_psi_grows_with_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=5000)
    small = mon.population_stability_index(ref, rng.normal(0.1, 1, 5000))
    large = mon.population_stability_index(ref, rng.normal(1.5, 1, 5000))
    assert 0 <= small < 0.1
    assert large > 0.25


def test_psi_is_non_negative_and_finite_with_empty_bin():
    # actual has nothing in the high tiers -> clipped 1e-6, must stay finite
    ref = np.array([0.0] * 90 + [0.9] * 10)
    act = np.zeros(100)
    psi = mon.population_stability_index(ref, act, bins=mon.RISK_SCORE_BINS)
    assert np.isfinite(psi) and psi > 0.25


def test_psi_zero_inflated_scores_with_fixed_bins_detects_shift():
    ref = np.array([0.0] * 94 + [0.5] * 6)
    same = np.array([0.0] * 94 + [0.5] * 6)
    shifted = np.array([0.0] * 70 + [0.5] * 30)
    assert mon.population_stability_index(ref, same, bins=mon.RISK_SCORE_BINS) == pytest.approx(0, abs=1e-9)
    assert mon.population_stability_index(ref, shifted, bins=mon.RISK_SCORE_BINS) > 0.25


def test_psi_does_not_mutate_bin_edges():
    edges = list(mon.RISK_SCORE_BINS)
    mon.population_stability_index([0.1, 0.2], [0.3], bins=edges)
    assert edges == mon.RISK_SCORE_BINS


def test_ks_same_vs_different():
    rng = np.random.default_rng(1)
    a = rng.normal(size=2000)
    stat_same, p_same = mon.ks_drift(a, a)
    stat_diff, p_diff = mon.ks_drift(a, a + 2)
    assert stat_same == 0.0 and p_same == pytest.approx(1.0)
    assert stat_diff > 0.5 and p_diff < 1e-6


def test_make_time_batches_weekly_chronological_and_complete():
    ts = pd.date_range("2020-01-01", periods=21, freq="D")  # 3 weeks
    df = pd.DataFrame({"trans_date_trans_time": ts, "v": range(21)})
    batches = mon.make_time_batches(df)
    assert sum(len(g) for _, g in batches) == 21
    labels = [b for b, _ in batches]
    assert labels == sorted(labels)
    assert len(batches) in (3, 4)


def test_batch_metrics_basic_and_drift_fields():
    y = np.array([1, 0, 0, 0, 1, 0])
    p = np.array([0.9, 0.1, 0.2, 0.0, 0.3, 0.0])
    m = mon.batch_metrics(
        y, p, threshold=0.5,
        reference_risk_scores=p, reference_amt=[1, 2, 3], batch_amt=[1, 2, 3],
    )
    assert m["n_transactions"] == 6
    assert m["n_alerts"] == 1
    assert m["recall"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(1.0)
    assert m["risk_score_psi"] == pytest.approx(0.0, abs=1e-9)
    assert m["amt_ks_stat"] == 0.0


def test_batch_metrics_omits_drift_without_references():
    m = mon.batch_metrics([1, 0, 0, 1], [0.9, 0.1, 0.2, 0.8], 0.5)
    assert "risk_score_psi" not in m and "amt_ks_stat" not in m


def test_monitor_runs_per_batch_and_indexes_by_period():
    ts = pd.date_range("2020-01-01", periods=28, freq="D")
    rng = np.random.default_rng(0)
    y = np.tile([1, 0, 0, 0], 7)
    df = pd.DataFrame({
        "trans_date_trans_time": ts, "is_fraud": y,
        "p": np.where(y == 1, 0.9, rng.uniform(0, 0.2, 28)),
        "amt": rng.uniform(1, 100, 28),
    })
    out = mon.monitor(df, "is_fraud", "p", "amt", 0.5,
                      reference_risk_scores=df["p"].values, reference_amt=df["amt"].values)
    assert out.index.name == "batch"
    assert out["n_transactions"].sum() == 28
    assert (out["risk_score_psi"] >= 0).all()
