# Claim-to-evidence table (CreditCardFraudDetection)

Evidence types: **README-reported** (original run, not reproduced), **regenerated** (recomputed from committed outputs), **test**, **CI**, **not run**. State on branch `improve/portfolio-refinement` (head 2ff0e88, CI run 43156061075 green). The data are a **simulated** Kaggle (Sparkov) dataset; nothing here is real bank data and no live monitoring exists.

| Claim | Status | Evidence type | Where | Caveat |
|---|---|---|---|---|
| Time-ordered split (training before 2020-05-10, validation 2020-05-10..06-21, held-out `fraudTest.csv`) | PARTIAL | README-reported | README Dataset | Raw CSVs not committed; training not rerun |
| Random Forest validation/test metrics at threshold 0.12 | VERIFIED (point estimates regenerate) | regenerated | `results/bootstrap_ci.md`, `scripts/bootstrap_report.py`, committed scored parquets | Precision@0.5% differs by one transaction (tie-breaking); models not retrained |
| 95% bootstrap CIs for RF metrics (row-level) | VERIFIED | regenerated, test | `results/bootstrap_ci.{md,json}`, `tests/test_bootstrap.py` | Row-level bootstrap treats transactions as independent |
| RF beats an amount-only baseline (paired PR-AUC difference) | VERIFIED | regenerated | `results/bootstrap_ci.md` | Weak baseline |
| RF beats XGBoost | UNSUPPORTED (not tested) | none | README states the paired comparison is not done | XGBoost scores not committed; needs the Kaggle data |
| Cost-sensitive threshold analysis | VERIFIED (method); results assumption-driven | test, regenerated | `src/cost.py`, `results/cost_sensitivity.md` | All costs are assumptions; optimal thresholds 0.04-0.44 vs README 0.12 |
| PSI/KS monitoring; weekly replay of `fraudTest.csv` (29 batches) | VERIFIED (functions); replay PARTIAL | test; README-reported | `tests/test_monitoring.py` | Retrospective replay, not live monitoring |
| Streamlit app renders all six pages under pinned dependencies | VERIFIED (local, headless) | test (opt-in `RUN_SLOW=1`) | `tests/test_app_smoke.py` | Not in CI; not a deployment |
| Hosted / deployed app | UNSUPPORTED | none | README links none | No deployment verified |
| Single-class batch bug in `fraud_metrics` fixed | VERIFIED | test | `tests/test_evaluate.py` | |
| 48 tests, ruff, CI | VERIFIED | CI | run 43156061075 | `app/app.py` and `build_deploy_assets.py` not unit tested |
| Rolling-origin validation | PROPOSED (blocked) | none | | Needs raw data and retraining |
