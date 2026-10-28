import numpy as np

from src.evaluate import bootstrap_macro_auroc, evaluate, macro_summary, per_label_metrics

LABELS = [
    {"id": "A", "tier": "primary"},
    {"id": "B", "tier": "primary"},
    {"id": "C", "tier": "primary"},
    {"id": "X", "tier": "exploratory"},
]


def _data():
    Y = np.array([[1, 0, 0, 1],
                  [0, 1, 0, 0],
                  [1, 0, 0, 0],
                  [0, 1, 0, 1]])
    P = np.array([[0.9, 0.2, 0.1, 0.1],     # A perfect, B perfect, X inverted (AUROC 0)
                  [0.1, 0.8, 0.3, 0.9],
                  [0.8, 0.1, 0.2, 0.8],
                  [0.2, 0.7, 0.4, 0.2]])
    return Y, P


def test_undefined_label_is_nan_and_flagged():
    Y, P = _data()
    pl = per_label_metrics(Y, P, LABELS)
    assert pl.loc["C", "undefined"] and np.isnan(pl.loc["C", "auroc"]) and np.isnan(pl.loc["C", "auprc"])
    assert not pl.loc["A", "undefined"]


def test_all_positive_label_is_undefined():
    Y, P = _data()
    Y[:, 1] = 1
    assert per_label_metrics(Y, P, LABELS).loc["B", "undefined"]


def test_macro_uses_primary_only_and_excludes_undefined():
    Y, P = _data()
    pl = per_label_metrics(Y, P, LABELS)
    assert pl.loc["X", "auroc"] == 0.0
    s = macro_summary(pl)
    assert s["macro_auroc_primary"] == 1.0          # mean of A and B; X (0.0) and C (NaN) excluded
    assert s["n_labels_in_macro"] == 2
    assert s["excluded_from_macro"] == ["C"]


def test_bootstrap_ci_contains_point_and_is_patient_level():
    rng = np.random.default_rng(0)
    n = 400
    Y = rng.integers(0, 2, size=(n, 4))
    P = Y * 0.6 + rng.random((n, 4)) * 0.8
    patients = np.repeat(np.arange(n // 2), 2)
    per_label, summary = evaluate(Y, P, LABELS, patient_ids=patients, n_boot=200)
    assert summary["ci_low"] <= summary["macro_auroc_primary"] <= summary["ci_high"]
    lo1, hi1 = bootstrap_macro_auroc(Y, P, LABELS, patients, n_boot=200, seed=0)
    lo2, hi2 = bootstrap_macro_auroc(Y, P, LABELS, patients, n_boot=200, seed=0)
    assert (lo1, hi1) == (lo2, hi2)
