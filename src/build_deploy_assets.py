"""One-off script that builds app/assets/ — the small, self-contained bundle
the Streamlit app needs to run on Streamlit Community Cloud without the full
raw CSVs or a from-scratch training run.

Run after regenerating models/ via the notebooks:

    python src/build_deploy_assets.py

Writes: app/assets/preprocessor.pkl, app/assets/random_forest.pkl (both
copied as-is), plus app/assets/val_scored.parquet and
app/assets/test_scored.parquet — the exact validation/test rows used
throughout this project, with engineered features and risk_score already
computed, so every number the deployed app shows matches the README and
notebooks exactly (this is precomputed real data, not a random resample).
"""

import shutil
from pathlib import Path

import joblib

import preprocessing as pp

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
ASSETS_DIR = ROOT_DIR / "app" / "assets"

KEEP_COLUMNS = [
    "trans_num",
    "trans_date_trans_time",
    "amt",
    "category",
    "gender",
    "day_of_week",
    "hour",
    "age",
    "city_pop",
    "merchant_distance_km",
    "is_fraud",
    "risk_score",
]


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    preprocessor = joblib.load(MODELS_DIR / "preprocessor.pkl")
    model = joblib.load(MODELS_DIR / "random_forest.pkl")
    shutil.copy(MODELS_DIR / "preprocessor.pkl", ASSETS_DIR / "preprocessor.pkl")
    shutil.copy(MODELS_DIR / "random_forest.pkl", ASSETS_DIR / "random_forest.pkl")

    train_raw = pp.engineer_features(pp.load_raw(DATA_DIR / "fraudTrain.csv"))
    _, val_split = pp.temporal_train_val_split(train_raw, val_weeks=6)
    val_split = val_split.reset_index(drop=True)
    X_val, _ = pp.split_X_y(val_split)
    val_split["risk_score"] = model.predict_proba(preprocessor.transform(X_val))[:, 1]
    val_split[KEEP_COLUMNS].to_parquet(ASSETS_DIR / "val_scored.parquet", index=False)

    test_raw = pp.engineer_features(pp.load_raw(DATA_DIR / "fraudTest.csv"))
    X_test, _ = pp.split_X_y(test_raw)
    test_raw["risk_score"] = model.predict_proba(preprocessor.transform(X_test))[:, 1]
    test_raw[KEEP_COLUMNS].to_parquet(ASSETS_DIR / "test_scored.parquet", index=False)

    for f in ASSETS_DIR.iterdir():
        print(f.name, f"{f.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
