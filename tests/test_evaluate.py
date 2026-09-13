import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import evaluate as ev  # noqa: E402


def test_fraud_metrics_known_confusion_matrix():
    # 2 TP, 1 FP, 1 FN, 6 TN -> precision=2/3, recall=2/3, fpr=1/7
    y_true = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
    y_proba = [0.9, 0.8, 0.2, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    m = ev.fraud_metrics(y_true, y_pred, y_proba)

    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(1 / 7)
    tn, fp, fn, tp = m["confusion_matrix"].ravel()
    assert (tn, fp, fn, tp) == (6, 1, 1, 2)


def test_fraud_metrics_no_predicted_positives_is_zero_not_nan():
    y_true = [1, 0, 0, 0]
    y_pred = [0, 0, 0, 0]
    y_proba = [0.3, 0.1, 0.1, 0.1]

    m = ev.fraud_metrics(y_true, y_pred, y_proba)

    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["fpr"] == 0.0


def test_metrics_table_structure():
    results = {
        "Model A": ev.fraud_metrics([1, 0], [1, 0], [0.9, 0.1]),
        "Model B": ev.fraud_metrics([1, 0], [0, 0], [0.4, 0.1]),
    }
    table = ev.metrics_table(results)

    assert list(table.index) == ["Model A", "Model B"]
    assert {"pr_auc", "roc_auc", "precision", "recall", "f1", "fpr"}.issubset(table.columns)


def test_precision_recall_at_k_perfect_ranking():
    # 10 transactions, 2 fraud, both ranked at the very top.
    y_true = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    y_proba = [0.99, 0.95, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]

    table = ev.precision_recall_at_k(y_true, y_proba, k_fracs=(0.1, 0.2))

    assert table.loc["Top 10%", "precision_at_k"] == pytest.approx(1.0)
    assert table.loc["Top 10%", "recall_at_k"] == pytest.approx(0.5)
    assert table.loc["Top 20%", "precision_at_k"] == pytest.approx(1.0)
    assert table.loc["Top 20%", "recall_at_k"] == pytest.approx(1.0)


def test_precision_recall_at_k_worst_ranking():
    # Fraud ranked dead last - top 20% should contain none of it.
    y_true = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    y_proba = [0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.1, 0.05]

    table = ev.precision_recall_at_k(y_true, y_proba, k_fracs=(0.2,))

    assert table.loc["Top 20%", "precision_at_k"] == pytest.approx(0.0)
    assert table.loc["Top 20%", "recall_at_k"] == pytest.approx(0.0)


def test_precision_recall_at_k_reports_review_counts():
    y_true = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    y_proba = list(range(10, 0, -1))  # 10 rows, strictly decreasing score

    table = ev.precision_recall_at_k(y_true, y_proba, k_fracs=(0.3,))

    assert table.loc["Top 30%", "n_reviewed"] == 3
