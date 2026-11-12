"""Cost-sensitive operating-point analysis on the committed RF scores.

Thresholds are CHOSEN on validation and EVALUATED on test (never tuned on test).
All cost parameters are assumptions (see src/cost.py); data is simulated.
Writes results/cost_sensitivity.md and results/cost_sensitivity.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from cost import CostParams, best_threshold, cost_curve  # noqa: E402

GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)
REVIEW = [1, 5, 15, 50]
FRICTION = [0, 2, 10]
RECOVERY = [0.0, 0.5]
CAPACITY_FRAC = 0.01  # reviewers can handle at most 1% of transactions (assumption)
README_THRESHOLD = 0.12


def per_1k(x, n):
    return 1000 * x / n


def main():
    val = pd.read_parquet(ROOT / "app/assets/val_scored.parquet")
    test = pd.read_parquet(ROOT / "app/assets/test_scored.parquet")
    args = lambda d: (d["is_fraud"].to_numpy(), d["risk_score"].to_numpy(), d["amt"].to_numpy())  # noqa: E731
    rows = []
    for rec in RECOVERY:
        for rv in REVIEW:
            for fr in FRICTION:
                prm = CostParams(rv, fr, rec)
                vc = cost_curve(*args(val), prm, GRID)
                tc = cost_curve(*args(test), prm, GRID).set_index("threshold")
                for cap_label, cap_val in (("none", None), (f"{CAPACITY_FRAC:.0%} of txns", CAPACITY_FRAC)):
                    b = best_threshold(vc, None if cap_val is None else int(cap_val * len(val)))
                    t = b["threshold"]
                    tr = tc.loc[t]
                    t_nothing = cost_curve(*args(test), prm, [1.1]).iloc[0]["total_cost"]
                    t_readme = tc.loc[README_THRESHOLD]
                    cap_ok = cap_val is None or tr["n_alerts"] <= cap_val * len(test) * 1.5
                    rows.append({
                        "recovery": rec, "review_cost": rv, "friction_cost": fr, "capacity": cap_label,
                        "val_best_threshold": t,
                        "test_alert_rate_pct": 100 * tr["n_alerts"] / len(test),
                        "test_cost_per_1k_at_best": per_1k(tr["total_cost"], len(test)),
                        "test_cost_per_1k_at_0.12": per_1k(t_readme["total_cost"], len(test)),
                        "test_cost_per_1k_alert_nothing": per_1k(t_nothing, len(test)),
                        "cap_ok": bool(cap_ok),
                    })
    df = pd.DataFrame(rows)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    df.to_json(out / "cost_sensitivity.json", orient="records", indent=2)
    unc = df[df["capacity"] == "none"]
    md = ["# Cost-sensitive threshold analysis (simulated data, assumption-driven)", "",
          "Random Forest scores from `app/assets/*_scored.parquet`. Threshold selected on validation, "
          "cost evaluated on the held-out test split. Costs are per 1,000 transactions in the dataset's "
          "dollars. **Review, friction and recovery values are assumptions, not measurements**; read the "
          "table as how the optimum moves, not as a loss estimate for any institution.", "",
          f"Compared with the README operating point (threshold {README_THRESHOLD}, max recall at FPR <= 0.5%) "
          "and with alerting nothing.", "",
          "## Optimal threshold by assumption (no capacity limit)", "",
          unc.drop(columns=["capacity", "cap_ok"]).round(3).to_markdown(index=False), "",
          f"## With a review-capacity cap of {CAPACITY_FRAC:.0%} of transactions (selected on validation)", "",
          df[df["capacity"] != "none"].drop(columns=["capacity"]).round(3).to_markdown(index=False), ""]
    (out / "cost_sensitivity.md").write_text("\n".join(md))
    print(unc["val_best_threshold"].describe())
    json.dump({"n_scenarios": len(df)}, sys.stdout)


if __name__ == "__main__":
    main()
