import numpy as np
from sklearn.metrics import roc_auc_score

from src import calibration as cal
from src.ensemble import apply_weights, cross_validated, fit_weights, group_folds


def _toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    Y = (rng.random((n, 3)) < [0.3, 0.1, 0.5]).astype(int)
    good = np.clip(Y * 0.6 + rng.random((n, 3)) * 0.5, 0, 1)
    noise = rng.random((n, 3))
    groups = np.repeat(np.arange(n // 3), 3)
    return Y, good, noise, groups


def test_group_folds_keep_patients_together():
    groups = np.repeat(np.arange(100), 3)
    for fit_idx, held_idx in group_folds(groups):
        assert not set(groups[fit_idx]) & set(groups[held_idx])


def test_weights_prefer_informative_member():
    Y, good, noise, _ = _toy()
    W = fit_weights([good, noise], Y)
    assert (W[:, 0] > W[:, 1]).all()
    assert np.allclose(W.sum(axis=1), 1)
    assert apply_weights([good, noise], W).shape == good.shape


def test_cross_validated_predictions_are_out_of_fold():
    Y, good, noise, groups = _toy()
    P = cross_validated([good, noise], Y, groups, "stacking")
    assert P.shape == Y.shape and np.isfinite(P).all()
    # stacking pure-noise members, scored out of fold, must stay near chance
    Pn = cross_validated([noise, noise[::-1]], Y, groups, "stacking")
    assert abs(roc_auc_score(Y[:, 0], Pn[:, 0]) - 0.5) < 0.1


def test_calibrators_roundtrip_and_temperature_identity():
    p = np.array([0.1, 0.4, 0.8])
    assert np.allclose(cal.apply_calibrator(p, {"method": "temperature", "T": 1.0}), p)
    assert np.allclose(cal.apply_calibrator(p, {"method": "platt", "a": 1.0, "b": 0.0}), p)
    assert np.allclose(cal.apply_calibrator(p, {"method": "none"}), p)


def test_platt_reduces_ece_on_miscalibrated_scores():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.2).astype(int)
    p_true = np.where(y == 1, 0.7, 0.1) + rng.normal(0, 0.05, len(y))
    p_bad = np.clip(p_true, 0.01, 0.99) ** 0.3          # pushed towards 1
    c = cal.fit_calibrator(p_bad, y, "platt")
    assert cal.ece(y, cal.apply_calibrator(p_bad, c)) < cal.ece(y, p_bad)


def test_sensitivity_threshold_meets_target():
    rng = np.random.default_rng(1)
    y = (rng.random(1000) < 0.1).astype(int)
    p = np.clip(y * 0.4 + rng.random(1000) * 0.6, 0, 1)
    t = cal.sens_threshold(p, y, 0.90)
    assert ((p >= t) & (y == 1)).sum() / y.sum() >= 0.90


def test_binary_metrics_macro_uses_primary_only():
    Y = np.array([[1, 0], [0, 1], [1, 0], [0, 0]])
    pred = np.array([[1, 1], [0, 0], [1, 1], [0, 1]])
    rows, agg = cal.binary_metrics(Y, pred, ["A", "X"], ["primary", "exploratory"])
    assert rows[0]["f1"] == 1.0 and rows[1]["f1"] == 0.0
    assert agg["macro_f1_primary"] == 1.0 and agg["micro_f1_primary"] == 1.0
