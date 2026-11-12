import numpy as np
import pytest

import cost


def _toy():
    # frauds: amt 1000 (score .9), 100 (score .3); legit: 4 rows
    y = np.array([1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.3, 0.5, 0.2, 0.1, 0.05])
    amt = np.array([1000.0, 100.0, 10.0, 10.0, 10.0, 10.0])
    return y, p, amt


def test_cost_curve_known_values():
    y, p, amt = _toy()
    prm = cost.CostParams(review_cost_per_alert=5, fp_friction_cost=2, recovery_rate=0.0)
    c = cost.cost_curve(y, p, amt, prm, [0.25]).iloc[0]
    # alerts: 0.9 (fraud), 0.3 (fraud), 0.5 (legit) -> n=3, tp=2, fp=1; nothing missed
    assert (c["n_alerts"], c["tp"], c["fp"]) == (3, 2, 1)
    assert c["missed_fraud_loss"] == 0
    assert c["total_cost"] == pytest.approx(3 * 5 + 1 * 2)


def test_alert_nothing_costs_the_full_fraud_amount_less_recovery():
    y, p, amt = _toy()
    c0 = cost.cost_curve(y, p, amt, cost.CostParams(recovery_rate=0.0), [1.1]).iloc[0]
    c5 = cost.cost_curve(y, p, amt, cost.CostParams(recovery_rate=0.5), [1.1]).iloc[0]
    assert c0["total_cost"] == pytest.approx(1100) and c5["total_cost"] == pytest.approx(550)


def test_best_threshold_trades_review_cost_against_missed_loss():
    y, p, amt = _toy()
    grid = [0.0, 0.25, 0.6, 0.95]
    cheap = cost.best_threshold(cost.cost_curve(y, p, amt, cost.CostParams(1, 0, 0), grid))
    pricey = cost.best_threshold(cost.cost_curve(y, p, amt, cost.CostParams(500, 0, 0), grid))
    assert cheap["threshold"] == 0.25          # catch both frauds, small review bill
    assert pricey["threshold"] == 0.6          # only the 1000 fraud is worth 500/alert...
    assert pricey["n_alerts"] <= cheap["n_alerts"]


def test_capacity_constraint_limits_alerts_and_can_be_infeasible():
    y, p, amt = _toy()
    curve = cost.cost_curve(y, p, amt, cost.CostParams(1, 0, 0), [0.0, 0.25, 0.6])
    assert cost.best_threshold(curve, max_alerts=1)["n_alerts"] <= 1
    assert cost.best_threshold(curve.iloc[[0]], max_alerts=1) is None


def test_sensitivity_grid_shape_and_monotone_alerts_in_review_cost():
    y, p, amt = _toy()
    g = cost.sensitivity_grid(y, p, amt, [0.0, 0.25, 0.6, 0.95], [1, 500], [0], [0.0])
    assert len(g) == 2
    low, high = g.sort_values("review_cost")["n_alerts"].tolist()
    assert high <= low
