"""Model construction helpers for Stage 4 benchmarking.

`build_models` returns plain (unweighted, unresampled) estimators.
`class_weight_ratio` and `apply_smote` support the imbalance-handling
comparison — resampling must only ever be applied to a training split,
never to validation/test.
"""

from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

RANDOM_STATE = 42


def class_weight_ratio(y_train) -> float:
    """neg/pos ratio, for XGBoost's scale_pos_weight."""
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    return n_neg / n_pos


def build_models(class_weighted: bool = False, scale_pos_weight: float = 1.0) -> dict:
    cw = "balanced" if class_weighted else None
    spw = scale_pos_weight if class_weighted else 1.0
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight=cw, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight=cw, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            scale_pos_weight=spw,
            eval_metric="logloss",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def apply_smote(Xt_train, y_train, sampling_strategy: float = 0.1):
    """Oversamples the training set only. sampling_strategy=0.1 means fraud
    becomes ~10% of the resampled training set, not a full 50/50 balance -
    at 0.578% real fraud, balancing fully would mean >99% of the minority
    class is synthetic. Even 10% is mostly synthetic; kept modest and
    compared empirically rather than assumed to help."""
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE)
    return smote.fit_resample(Xt_train, y_train)
