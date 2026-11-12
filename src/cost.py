"""Cost-sensitive threshold analysis. ALL COST PARAMETERS ARE ASSUMPTIONS.

The data is simulated: there are no real chargeback, recovery or investigation costs
here, so the numbers below show how the optimal operating point MOVES with the
assumptions; they are not estimates of any institution's losses.

Per-period cost of a threshold t (alerts = score >= t):
    missed_fraud_loss  = sum(amt of frauds not alerted) * (1 - recovery_rate)
    review_cost        = review_cost_per_alert * n_alerts
    friction_cost      = fp_friction_cost * n_false_positive_alerts
A caught fraud costs only its review. total = sum of the three.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostParams:
    review_cost_per_alert: float = 5.0   # analyst time per reviewed alert (assumption)
    fp_friction_cost: float = 2.0        # customer friction per false alert (assumption)
    recovery_rate: float = 0.0           # share of a MISSED fraud's amount recovered (assumption)


def cost_curve(y, p, amt, params: CostParams, thresholds) -> pd.DataFrame:
    y = np.asarray(y)
    p = np.asarray(p)
    amt = np.asarray(amt, dtype=float)
    total_fraud_amt = float(amt[y == 1].sum())
    rows = []
    for t in thresholds:
        alert = p >= t
        n_alerts = int(alert.sum())
        tp = int((alert & (y == 1)).sum())
        fp = n_alerts - tp
        caught_amt = float(amt[alert & (y == 1)].sum())
        missed = (total_fraud_amt - caught_amt) * (1 - params.recovery_rate)
        review = params.review_cost_per_alert * n_alerts
        friction = params.fp_friction_cost * fp
        rows.append({"threshold": float(t), "n_alerts": n_alerts, "tp": tp, "fp": fp,
                     "missed_fraud_loss": missed, "review_cost": review,
                     "friction_cost": friction, "total_cost": missed + review + friction})
    return pd.DataFrame(rows)


def best_threshold(curve: pd.DataFrame, max_alerts: int | None = None):
    """Lowest-total-cost row, optionally restricted to n_alerts <= max_alerts
    (review capacity). None if no row is feasible."""
    c = curve if max_alerts is None else curve[curve["n_alerts"] <= max_alerts]
    if c.empty:
        return None
    return c.loc[c["total_cost"].idxmin()]


def sensitivity_grid(y, p, amt, thresholds, review_costs, friction_costs, recovery_rates,
                     capacity_alerts=None) -> pd.DataFrame:
    """Optimal threshold under every combination of assumptions."""
    rows = []
    for r in review_costs:
        for f in friction_costs:
            for rec in recovery_rates:
                params = CostParams(r, f, rec)
                curve = cost_curve(y, p, amt, params, thresholds)
                best = best_threshold(curve, capacity_alerts)
                if best is None:
                    continue
                rows.append({"review_cost": r, "friction_cost": f, "recovery_rate": rec,
                             "best_threshold": best["threshold"], "n_alerts": int(best["n_alerts"]),
                             "total_cost": best["total_cost"]})
    return pd.DataFrame(rows)
