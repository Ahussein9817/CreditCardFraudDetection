"""Streamlit fraud-monitoring dashboard.

Loads the Random Forest model and preprocessor persisted in Stage 5,
scores the validation split and the held-out test set (Stage 10's
monitoring simulation), and exposes six pages: Overview, Risk
Distribution, Investigator Queue, Threshold Simulator, Model
Monitoring, and Explainability.
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))
import evaluate as ev  # noqa: E402
import monitoring as mon  # noqa: E402
import preprocessing as pp  # noqa: E402

DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
ASSETS_DIR = ROOT_DIR / "app" / "assets"
DEFAULT_THRESHOLD = 0.12  # chosen in Stage 5

# app/assets/ (committed, via src/build_deploy_assets.py) is a small,
# self-contained bundle for cloud deployment - the full data/ and models/
# directories are gitignored (raw CSVs + a 70MB model) and only exist
# after running the notebooks locally. Prefer assets/ when present.
USE_PREBUILT_ASSETS = (ASSETS_DIR / "preprocessor.pkl").exists()

sns.set_style("whitegrid")
st.set_page_config(page_title="Fraud Monitoring Dashboard", layout="wide")


# ---------------------------------------------------------------------------
# Cached data / model loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model_and_preprocessor():
    source_dir = ASSETS_DIR if USE_PREBUILT_ASSETS else MODELS_DIR
    preprocessor = joblib.load(source_dir / "preprocessor.pkl")
    model = joblib.load(source_dir / "random_forest.pkl")
    return preprocessor, model


@st.cache_resource
def get_shap_explainer():
    _, model = load_model_and_preprocessor()
    return shap.TreeExplainer(model)


@st.cache_data
def get_scored_validation() -> pd.DataFrame:
    if USE_PREBUILT_ASSETS:
        return pd.read_parquet(ASSETS_DIR / "val_scored.parquet")

    preprocessor, model = load_model_and_preprocessor()
    train_raw = pp.engineer_features(pp.load_raw(DATA_DIR / "fraudTrain.csv"))
    _, val_split = pp.temporal_train_val_split(train_raw, val_weeks=6)
    val_split = val_split.reset_index(drop=True)
    X_val, _ = pp.split_X_y(val_split)
    val_split = val_split.copy()
    val_split["risk_score"] = model.predict_proba(preprocessor.transform(X_val))[:, 1]
    return val_split


@st.cache_data
def get_val_transformed() -> np.ndarray:
    preprocessor, _ = load_model_and_preprocessor()
    val_scored = get_scored_validation()
    X_val, _ = pp.split_X_y(val_scored)
    return preprocessor.transform(X_val).toarray()


@st.cache_data
def get_monitoring_report() -> pd.DataFrame:
    val_scored = get_scored_validation()

    if USE_PREBUILT_ASSETS:
        test_scored = pd.read_parquet(ASSETS_DIR / "test_scored.parquet")
    else:
        preprocessor, model = load_model_and_preprocessor()
        test_scored = pp.engineer_features(pp.load_raw(DATA_DIR / "fraudTest.csv")).copy()
        X_test, _ = pp.split_X_y(test_scored)
        test_scored["risk_score"] = model.predict_proba(preprocessor.transform(X_test))[:, 1]

    return mon.monitor(
        test_scored,
        y_true_col="is_fraud",
        proba_col="risk_score",
        amt_col="amt",
        threshold=DEFAULT_THRESHOLD,
        reference_risk_scores=val_scored["risk_score"].values,
        reference_amt=val_scored["amt"].values,
        freq="W",
    )


@st.cache_data
def get_shap_global_sample():
    preprocessor, _ = load_model_and_preprocessor()
    val_scored = get_scored_validation()
    Xt_val = get_val_transformed()
    feature_names = pp.get_feature_names(preprocessor)

    rng = np.random.default_rng(42)
    fraud_idx = np.where(val_scored["is_fraud"].values == 1)[0]
    legit_idx = np.where(val_scored["is_fraud"].values == 0)[0]
    sample_idx = np.concatenate(
        [rng.choice(fraud_idx, size=200, replace=False), rng.choice(legit_idx, size=200, replace=False)]
    )
    sample = Xt_val[sample_idx]

    explainer = get_shap_explainer()
    shap_values = explainer.shap_values(sample)[:, :, 1]
    return sample, shap_values, feature_names


def explain_row(position: int, feature_names) -> pd.Series:
    explainer = get_shap_explainer()
    Xt_val = get_val_transformed()
    row_shap = explainer.shap_values(Xt_val[position : position + 1])[:, :, 1][0]
    return pd.Series(row_shap, index=feature_names).sort_values(key=abs, ascending=False)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_overview():
    st.title("Fraud Monitoring Dashboard")
    st.caption(f"Random Forest, operating threshold {DEFAULT_THRESHOLD} (chosen in Stage 5), evaluated on the validation split.")

    val_scored = get_scored_validation()
    pred = (val_scored["risk_score"] >= DEFAULT_THRESHOLD).astype(int)
    m = ev.fraud_metrics(val_scored["is_fraud"], pred, val_scored["risk_score"])
    n = len(val_scored)
    n_alerts = int(pred.sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions processed", f"{n:,}")
    col2.metric("Alerts generated", f"{n_alerts:,}", f"{100 * n_alerts / n:.2f}% flagged")
    col3.metric("Observed fraud rate", f"{100 * val_scored['is_fraud'].mean():.3f}%")
    col4.metric("PR-AUC", f"{m['pr_auc']:.3f}")

    col5, col6, col7 = st.columns(3)
    col5.metric("Precision", f"{m['precision']:.3f}")
    col6.metric("Recall", f"{m['recall']:.3f}")
    col7.metric("False Positive Rate", f"{m['fpr']:.4f}")

    st.divider()
    st.caption(
        "On the truly held-out test period (Stage 10), this same threshold reaches "
        "precision 0.424 / recall 0.909 / FPR 0.0048 - recall and FPR hold up almost "
        "exactly, precision degrades somewhat on genuinely new data. See the Model "
        "Monitoring page."
    )


def page_risk_distribution():
    st.title("Risk Score Distribution")
    val_scored = get_scored_validation()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.histplot(
        val_scored.loc[val_scored["is_fraud"] == 0, "risk_score"],
        bins=50, ax=ax, color="steelblue", label="Legitimate", stat="density", alpha=0.5,
    )
    sns.histplot(
        val_scored.loc[val_scored["is_fraud"] == 1, "risk_score"],
        bins=50, ax=ax, color="indianred", label="Fraud", stat="density", alpha=0.5,
    )
    ax.axvline(DEFAULT_THRESHOLD, color="black", linestyle="--", label=f"Current threshold ({DEFAULT_THRESHOLD})")
    ax.set_xlabel("Predicted fraud probability")
    ax.set_yscale("log")
    ax.set_ylabel("Density (log scale)")
    ax.legend()
    st.pyplot(fig)

    pct_zero = 100 * (val_scored["risk_score"] == 0).mean()
    st.caption(
        f"{pct_zero:.1f}% of transactions score exactly 0.0 - this Random Forest's "
        "scores are heavily zero-inflated (log scale used above so the fraud "
        "distribution stays visible). See Stage 10 for why this required a "
        "fixed-bin PSI calculation rather than quantile bins."
    )


def page_investigator_queue():
    st.title("Investigator Review Queue")
    val_scored = get_scored_validation()

    capacity_pct = st.select_slider(
        "Investigator review capacity", options=[0.5, 1, 2, 5, 10], value=1.0,
        format_func=lambda x: f"{x}%",
    )
    k = capacity_pct / 100
    n = len(val_scored)
    top_n = max(1, int(round(n * k)))

    queue = val_scored.sort_values("risk_score", ascending=False).reset_index(drop=True)
    top = queue.head(top_n)
    fraud_captured = int(top["is_fraud"].sum())
    total_fraud = int(val_scored["is_fraud"].sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Transactions reviewed", f"{top_n:,}")
    col2.metric(
        "Fraud captured", f"{fraud_captured} / {total_fraud}",
        f"{100 * fraud_captured / total_fraud:.1f}% recall",
    )
    col3.metric("Precision in this queue", f"{100 * fraud_captured / top_n:.1f}%")

    show = top[["trans_num", "amt", "category", "risk_score", "is_fraud"]].copy()
    show["prediction"] = np.where(show["risk_score"] >= DEFAULT_THRESHOLD, "Fraud", "Legitimate")
    show["actual"] = np.where(show["is_fraud"] == 1, "Fraud", "Legitimate")
    show = show.drop(columns="is_fraud").rename(
        columns={"trans_num": "Transaction", "amt": "Amount ($)", "category": "Category", "risk_score": "Risk Score"}
    )
    st.dataframe(show, width="stretch", height=500)


def page_threshold_simulator():
    st.title("Threshold Simulator")
    val_scored = get_scored_validation()

    threshold = st.slider("Risk threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01)
    pred = (val_scored["risk_score"] >= threshold).astype(int)
    m = ev.fraud_metrics(val_scored["is_fraud"], pred, val_scored["risk_score"])
    n = len(val_scored)
    n_alerts = int(pred.sum())
    tp = int(((val_scored["is_fraud"] == 1) & (pred == 1)).sum())
    fn = int(((val_scored["is_fraud"] == 1) & (pred == 0)).sum())
    fp = int(((val_scored["is_fraud"] == 0) & (pred == 1)).sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Fraud captured", tp)
    col2.metric("Fraud missed", fn)
    col3.metric("False positives", fp)
    col4.metric("Alerts generated", f"{n_alerts:,}")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Precision", f"{m['precision']:.3f}")
    col6.metric("Recall", f"{m['recall']:.3f}")
    col7.metric("False Positive Rate", f"{m['fpr']:.4f}")
    col8.metric("% of transactions flagged", f"{100 * n_alerts / n:.2f}%")

    st.caption("Stage 5 chose 0.12 to maximize recall subject to FPR <= 0.5%. Move the slider to see the tradeoff directly.")

    st.subheader("Confusion matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        m["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax_cm,
        xticklabels=["Predicted legit", "Predicted fraud"],
        yticklabels=["Actual legit", "Actual fraud"],
    )
    ax_cm.set_title(f"Threshold = {threshold:.2f}")
    st.pyplot(fig_cm)
    plt.close(fig_cm)


def page_monitoring():
    st.title("Model Monitoring")
    st.caption(
        "fraudTest.csv (2020-06-21 to 2020-12-31), replayed as weekly batches, "
        "compared against the validation set the model was tuned on (Stage 10)."
    )

    with st.spinner("Scoring the held-out test set and computing batch metrics..."):
        report = get_monitoring_report()

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    x = range(len(report))
    axes[0].plot(x, report["precision"], marker="o", markersize=3, label="Precision")
    axes[0].plot(x, report["recall"], marker="o", markersize=3, label="Recall")
    axes[0].plot(x, report["pr_auc"], marker="o", markersize=3, label="PR-AUC")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Fraud performance by batch")

    axes[1].plot(x, report["risk_score_psi"], marker="o", markersize=3, color="purple", label="Risk-score PSI")
    axes[1].axhline(0.1, linestyle="--", color="gray", linewidth=0.8, label="PSI = 0.1 (moderate-shift guideline)")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Risk-score drift (PSI vs. validation reference)")
    axes[1].set_xlabel("Batch (week, chronological)")

    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(
        report[["n_transactions", "pct_flagged", "observed_fraud_rate_pct", "precision", "recall", "pr_auc", "risk_score_psi"]].round(4),
        width="stretch",
    )
    st.caption(
        "The final batch is a partial 4-day tail with zero observed fraud - its "
        "precision/recall/PR-AUC show as 0.0 because those metrics are undefined "
        "with no positive class present, not because the model failed."
    )


def page_explainability():
    st.title("Model Explainability")
    st.caption("Every feature here is genuinely interpretable (not anonymized PCA), so these explanations translate directly into investigator-facing reasoning.")

    with st.spinner("Computing SHAP values (first load only, roughly 1-2 minutes)..."):
        sample, shap_values, feature_names = get_shap_global_sample()

    st.subheader("Global feature importance")
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_abs.tail(12).plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Mean |SHAP value| (impact on fraud-risk score)")
    st.pyplot(fig)

    st.subheader("SHAP summary")
    fig2 = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False, max_display=12)
    st.pyplot(fig2)
    plt.close(fig2)

    st.subheader("Explain an individual alert")
    val_scored = get_scored_validation()
    top_alerts = val_scored[val_scored["risk_score"] >= DEFAULT_THRESHOLD].sort_values("risk_score", ascending=False).head(50)
    options = top_alerts.index.tolist()

    def _label(i):
        row = top_alerts.loc[i]
        return f"{row['trans_num'][:10]}...  |  ${row['amt']:.2f}  |  {row['category']}  |  score={row['risk_score']:.3f}"

    selected = st.selectbox("Pick a flagged transaction", options, format_func=_label)
    contributions = explain_row(selected, feature_names)
    row = val_scored.loc[selected]

    col1, col2, col3 = st.columns(3)
    col1.metric("Risk score", f"{row['risk_score']:.3f}")
    col2.metric("Predicted", "Fraud" if row["risk_score"] >= DEFAULT_THRESHOLD else "Legitimate")
    col3.metric("Actual", "Fraud" if row["is_fraud"] == 1 else "Legitimate")

    st.write("Top contributing features:")
    st.dataframe(contributions.head(8).rename("SHAP contribution").to_frame())


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

PAGES = {
    "Overview": page_overview,
    "Risk Distribution": page_risk_distribution,
    "Investigator Queue": page_investigator_queue,
    "Threshold Simulator": page_threshold_simulator,
    "Model Monitoring": page_monitoring,
    "Explainability": page_explainability,
}

st.sidebar.title("Fraud Monitoring")
selected_page = st.sidebar.radio("Navigate", list(PAGES.keys()))
PAGES[selected_page]()
