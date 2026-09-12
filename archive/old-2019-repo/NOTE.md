# Original 2019 Iteration — Archived

This folder preserves the original version of this project (notebook,
README, and result images) for historical reference. It is not part of the
current pipeline.

## Why it was rebuilt

- **Dataset no longer available.** The original ULB/Kaggle anonymized
  dataset (`V1`-`V28` PCA features, 284,807 transactions) used here was
  never committed to the repo and could not be recovered locally. It's
  still publicly available at
  [kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
  if needed for reference, but the rebuild uses a different, richer dataset
  (see main README).
- **Data leakage.** SMOTE was applied in a way that leaked information into
  evaluation, producing the "perfect" Decision Tree scores in
  `Results/DecisionTreeonBalancedData.png` — a classic overfitting artifact,
  not a real result. The rebuilt pipeline fits all resampling and
  preprocessing strictly on the training split only.
- **Anonymized features blocked half the project.** With only `V1`-`V28`,
  there's no way to build interpretable fraud rules or give investigators
  a real explanation for a flagged transaction. The rebuild uses a dataset
  with genuine interpretable features (merchant, category, amount,
  demographics, geolocation, timestamps), enabling real rules and
  explainability.

See the top-level `README.md` for the current project.
