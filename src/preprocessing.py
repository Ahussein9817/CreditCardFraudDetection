"""Feature engineering and preprocessing for the credit-card fraud dataset.

Fitting (scaling/encoding) must always happen on the training split only —
call `build_preprocessor()` and `fit_transform` on train, then `transform`
(never `fit`) on validation/test.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = ["amt", "city_pop", "age", "merchant_distance_km", "hour"]
CATEGORICAL_FEATURES = ["category", "gender", "day_of_week"]
TARGET = "is_fraud"


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds derived columns. Safe to call on train/val/test independently —
    every derived value depends only on that row's own fields, never on
    dataset-wide statistics (those go through build_preprocessor instead)."""
    df = df.copy()
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])

    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day_of_week"] = df["trans_date_trans_time"].dt.day_name()
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days // 365
    df["merchant_distance_km"] = haversine_km(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )
    return df


def temporal_train_val_split(df: pd.DataFrame, val_weeks: int = 6):
    """Splits a chronologically-sorted-or-not dataframe into train/validation
    by time: the last `val_weeks` of the period become validation, everything
    before that is train. Requires `trans_date_trans_time` to already be
    parsed as datetime (i.e. call after `engineer_features`)."""
    cutoff = df["trans_date_trans_time"].max() - pd.Timedelta(weeks=val_weeks)
    train = df[df["trans_date_trans_time"] <= cutoff].copy()
    val = df[df["trans_date_trans_time"] > cutoff].copy()
    return train, val


def get_feature_columns():
    return list(NUMERIC_FEATURES), list(CATEGORICAL_FEATURES)


def build_preprocessor() -> ColumnTransformer:
    """Returns an unfit ColumnTransformer. Fit only on the training split."""
    numeric_pipeline = Pipeline([("scale", StandardScaler())])
    categorical_pipeline = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def get_feature_names(preprocessor: ColumnTransformer):
    """Feature names out of a *fitted* preprocessor, in transform-output order."""
    return list(preprocessor.get_feature_names_out())


def split_X_y(df: pd.DataFrame):
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    return df[feature_cols], df[TARGET]
