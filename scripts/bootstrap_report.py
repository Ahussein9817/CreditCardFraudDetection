"""Bootstrap CIs for the committed Random Forest scores (validation and held-out test).

Inputs: app/assets/{val,test}_scored.parquet (committed; columns is_fraud, risk_score, amt).
Output: results/bootstrap_ci.json and results/bootstrap_ci.md

Threshold 0.12 is the validation-chosen operating point from the README (max recall at
FPR <= 0.5%); it is NOT re-tuned on test. Paired comparison is Random Forest vs an
amount-only ranking baseline (score = amt rank). A Random Forest vs XGBoost comparison
needs XGBoost scores that are not committed (they require the Kaggle data and retraining).
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from bootstrap import bootstrap_metrics, paired_bootstrap_diff  # noqa: E402

THRESHOLD = 0.12
N_VAL, N_TEST = 1000, 300


def main():
    res = {"threshold": THRESHOLD, "splits": {}}
    for split, n_boot in (("val", N_VAL), ("test", N_TEST)):
        d = pd.read_parquet(ROOT / f"app/assets/{split}_scored.parquet")
        y, p = d["is_fraud"].to_numpy(), d["risk_score"].to_numpy()
        amt_score = d["amt"].rank(pct=True).to_numpy()
        r = bootstrap_metrics(y, p, THRESHOLD, n_boot=n_boot, seed=42)
        r["n_rows"], r["n_fraud"] = int(len(d)), int(y.sum())
        r["paired_vs_amount_only"] = {
            m: paired_bootstrap_diff(y, p, amt_score, metric=m, n_boot=min(n_boot, 200), seed=42)
            for m in ("pr_auc", "roc_auc")
        }
        res["splits"][split] = r
        print(split, "done", flush=True)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "bootstrap_ci.json").write_text(json.dumps(res, indent=2))
    lines = ["# Bootstrap 95% CIs, Random Forest (committed scores)", "",
             f"Threshold {THRESHOLD} (validation-chosen). Row-level bootstrap, seed 42; "
             "no cardholder id is available so correlation within cards is ignored.", ""]
    for split, r in res["splits"].items():
        lines += [f"## {split}: {r['n_rows']:,} rows, {r['n_fraud']} frauds, {r['n_boot']} resamples "
                  f"({r['n_skipped']} skipped)", "", "| metric | point | 95% CI |", "|---|---|---|"]
        for k, v in r["metrics"].items():
            lines.append(f"| {k} | {v['point']:.4f} | [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}] |")
        lines.append("")
        for m, d in r["paired_vs_amount_only"].items():
            lines.append(f"Paired RF minus amount-only baseline, {m}: {d['diff_point']:+.4f} "
                         f"CI [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}] ({d['n_boot']} resamples)")
        lines.append("")
    (out / "bootstrap_ci.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
