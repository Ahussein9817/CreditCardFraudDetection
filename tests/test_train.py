import numpy as np
import pytest

import train


def test_class_weight_ratio():
    y = np.array([1, 0, 0, 0, 0])
    assert train.class_weight_ratio(y) == pytest.approx(4.0)


def test_build_models_unweighted_and_weighted():
    plain = train.build_models()
    assert set(plain) == {"Logistic Regression", "Random Forest", "XGBoost"}
    assert plain["Logistic Regression"].class_weight is None
    assert plain["XGBoost"].scale_pos_weight == 1.0

    weighted = train.build_models(class_weighted=True, scale_pos_weight=42.0)
    assert weighted["Random Forest"].class_weight == "balanced"
    assert weighted["XGBoost"].scale_pos_weight == 42.0


def test_build_models_fit_and_predict_proba_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = (X[:, 0] > 1).astype(int)
    for name, model in train.build_models(class_weighted=True, scale_pos_weight=4.0).items():
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (200, 2), name
        assert np.allclose(proba.sum(axis=1), 1.0), name


def test_apply_smote_only_grows_minority_to_target_ratio():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, 3))
    y = np.zeros(1000, dtype=int)
    y[:20] = 1
    X[:20] += 3
    Xr, yr = train.apply_smote(X, y, sampling_strategy=0.1)
    assert (yr == 0).sum() == 980  # majority untouched
    assert (yr == 1).sum() == 98  # 0.1 * 980
    assert Xr.shape[0] == len(yr)
