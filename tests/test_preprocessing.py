import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import preprocessing as pp  # noqa: E402


def make_raw_df():
    return pd.DataFrame(
        {
            "trans_date_trans_time": [
                "2020-01-01 10:00:00",
                "2020-01-15 23:30:00",
                "2020-02-01 03:00:00",
            ],
            "dob": ["1990-01-01", "1985-06-15", "2000-12-31"],
            "lat": [40.7128, 34.0522, 51.5074],
            "long": [-74.0060, -118.2437, -0.1278],
            "merch_lat": [40.7128, 34.5, 48.8566],
            "merch_long": [-74.0060, -118.5, 2.3522],
            "amt": [10.0, 200.0, 55.5],
            "category": ["grocery_pos", "shopping_net", "gas_transport"],
            "gender": ["F", "M", "F"],
            "city_pop": [1000, 50000, 2000],
            "is_fraud": [0, 1, 0],
        }
    )


def test_haversine_km_zero_distance():
    assert pp.haversine_km(40.0, -74.0, 40.0, -74.0) == pytest.approx(0.0, abs=1e-6)


def test_haversine_km_known_distance():
    # NYC to LA, commonly cited great-circle distance ~3936 km.
    d = pp.haversine_km(40.7128, -74.0060, 34.0522, -118.2437)
    assert d == pytest.approx(3936, rel=0.01)


def test_engineer_features_adds_expected_columns():
    df = pp.engineer_features(make_raw_df())
    for col in ["hour", "day_of_week", "age", "merchant_distance_km"]:
        assert col in df.columns

    assert df["hour"].tolist() == [10, 23, 3]
    assert df.loc[0, "age"] == 30  # exactly 30 years, 1990-01-01 -> 2020-01-01
    assert df.loc[0, "merchant_distance_km"] == pytest.approx(0.0, abs=1e-6)
    assert (df["age"] >= 0).all()


def test_engineer_features_does_not_mutate_input():
    raw = make_raw_df()
    raw_copy = raw.copy()
    pp.engineer_features(raw)
    pd.testing.assert_frame_equal(raw, raw_copy)


def test_temporal_train_val_split_no_overlap_and_full_coverage():
    df = pp.engineer_features(make_raw_df())
    train, val = pp.temporal_train_val_split(df, val_weeks=2)
    assert len(train) + len(val) == len(df)
    if len(train) and len(val):
        assert train["trans_date_trans_time"].max() < val["trans_date_trans_time"].min()


def test_temporal_train_val_split_all_val_when_val_weeks_huge():
    # A huge val_weeks pushes the cutoff far into the past, so every row
    # (being "after" that cutoff) falls into val, none into train.
    df = pp.engineer_features(make_raw_df())
    train, val = pp.temporal_train_val_split(df, val_weeks=1000)
    assert len(train) == 0
    assert len(val) == len(df)


def test_split_X_y_returns_expected_columns_and_target():
    df = pp.engineer_features(make_raw_df())
    X, y = pp.split_X_y(df)
    numeric, categorical = pp.get_feature_columns()
    assert list(X.columns) == numeric + categorical
    assert y.tolist() == [0, 1, 0]


def test_build_preprocessor_fit_transform_shape():
    df = pp.engineer_features(make_raw_df())
    X, _ = pp.split_X_y(df)
    preprocessor = pp.build_preprocessor()
    Xt = preprocessor.fit_transform(X)
    feature_names = pp.get_feature_names(preprocessor)
    assert Xt.shape == (len(df), len(feature_names))


def test_build_preprocessor_handles_unseen_category_without_raising():
    df = pp.engineer_features(make_raw_df())
    X, _ = pp.split_X_y(df)
    preprocessor = pp.build_preprocessor()
    preprocessor.fit(X)

    unseen = X.copy()
    unseen.loc[0, "category"] = "totally_new_category_not_in_train"
    Xt = preprocessor.transform(unseen)  # would raise without handle_unknown="ignore"
    assert Xt.shape[0] == len(unseen)
