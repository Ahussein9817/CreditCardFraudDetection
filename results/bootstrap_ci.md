# Bootstrap 95% CIs, Random Forest (committed scores)

Threshold 0.12 (validation-chosen). Row-level bootstrap, seed 42; no cardholder id is available so correlation within cards is ignored.

## val: 111,581 rows, 669 frauds, 1000 resamples (0 skipped)

| metric | point | 95% CI |
|---|---|---|
| pr_auc | 0.9126 | [0.8935, 0.9306] |
| roc_auc | 0.9918 | [0.9869, 0.9960] |
| precision | 0.5391 | [0.5096, 0.5672] |
| recall | 0.9178 | [0.8977, 0.9383] |
| fpr | 0.0047 | [0.0043, 0.0051] |
| precision_at_0.5pct | 0.9516 | [0.9104, 0.9821] |
| recall_at_0.5pct | 0.7937 | [0.7587, 0.8291] |
| precision_at_1pct | 0.5502 | [0.5063, 0.5896] |
| recall_at_1pct | 0.9178 | [0.8982, 0.9379] |

Paired RF minus amount-only baseline, pr_auc: +0.7096 CI [+0.6858, +0.7340] (200 resamples)
Paired RF minus amount-only baseline, roc_auc: +0.1549 CI [+0.1337, +0.1768] (200 resamples)

## test: 555,719 rows, 2145 frauds, 300 resamples (0 skipped)

| metric | point | 95% CI |
|---|---|---|
| pr_auc | 0.8767 | [0.8652, 0.8900] |
| roc_auc | 0.9868 | [0.9838, 0.9898] |
| precision | 0.4235 | [0.4096, 0.4381] |
| recall | 0.9091 | [0.8978, 0.9210] |
| fpr | 0.0048 | [0.0046, 0.0050] |
| precision_at_0.5pct | 0.6653 | [0.6378, 0.6981] |
| recall_at_0.5pct | 0.8620 | [0.8501, 0.8773] |
| precision_at_1pct | 0.3579 | [0.3429, 0.3741] |
| recall_at_1pct | 0.9273 | [0.9167, 0.9373] |

Paired RF minus amount-only baseline, pr_auc: +0.7398 CI [+0.7300, +0.7530] (200 resamples)
Paired RF minus amount-only baseline, roc_auc: +0.1536 CI [+0.1430, +0.1649] (200 resamples)
