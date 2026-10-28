"""
Per-label probability calibration and decision thresholds, estimated inside fold 9 with 5-fold
patient-grouped cross-validation (fit on 4/5, apply to the held-out 1/5, pool).

Calibrators (on the logit z of the model probability):
  none         p
  temperature  sigmoid(z / T)
  platt        sigmoid(a * z + b)
Per label the method with the lowest cross-validated log-loss is chosen.

Threshold policies (on calibrated probabilities):
  max_f1       threshold maximising F1 on the fitting part
  sens90       for high-risk labels: the highest threshold with sensitivity >= 0.90 on the fitting
               part; other labels use max_f1
"""
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from .ensemble import group_folds, logit

METHODS = ["none", "temperature", "platt"]
HIGH_RISK = ["AFIB", "AFLT", "SVT", "AVB_HIGH", "LBBB"]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_calibrator(p, y, method):
    z = logit(p)
    if method == "none" or not (0 < y.sum() < len(y)):
        return {"method": "none"}
    if method == "temperature":
        res = minimize_scalar(lambda t: log_loss(y, np.clip(_sigmoid(z / t), 1e-7, 1 - 1e-7),
                                                 labels=[0, 1]),
                              bounds=(0.05, 20.0), method="bounded")
        return {"method": "temperature", "T": float(res.x)}
    if method == "platt":
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(z[:, None], y)
        return {"method": "platt", "a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}
    raise ValueError(method)


def apply_calibrator(p, cal):
    z = logit(p)
    if cal["method"] == "temperature":
        return _sigmoid(z / cal["T"])
    if cal["method"] == "platt":
        return _sigmoid(cal["a"] * z + cal["b"])
    return np.asarray(p, dtype=float)


def ece(y, p, n_bins=15):
    """Expected calibration error with equal-width bins."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def cv_calibrate(P, Y, groups, n_splits=5, seed=0):
    """Choose a method per label by CV log-loss. Returns (pooled CV-calibrated P, chosen methods, cv losses)."""
    folds = group_folds(groups, n_splits, seed)
    out = {m: np.zeros_like(P) for m in METHODS}
    for fit_idx, held_idx in folds:
        for j in range(P.shape[1]):
            for m in METHODS:
                cal = fit_calibrator(P[fit_idx, j], Y[fit_idx, j], m)
                out[m][held_idx, j] = apply_calibrator(P[held_idx, j], cal)
    chosen, losses = [], []
    P_cal = np.zeros_like(P)
    for j in range(P.shape[1]):
        ll = {m: log_loss(Y[:, j], np.clip(out[m][:, j], 1e-7, 1 - 1e-7), labels=[0, 1]) for m in METHODS}
        best = min(ll, key=ll.get)
        chosen.append(best)
        losses.append(ll)
        P_cal[:, j] = out[best][:, j]
    return P_cal, chosen, losses


def best_f1_threshold(p, y):
    order = np.argsort(-p)
    ps, ys = p[order], y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    f1 = 2 * tp / (tp + fp + ys.sum())
    k = int(np.argmax(f1))
    return float(ps[k])


def sens_threshold(p, y, target=0.90):
    pos = np.sort(p[y == 1])
    if len(pos) == 0:
        return 0.5
    k = int(np.floor((1 - target) * len(pos)))      # keep >= target fraction of positives at/above
    return float(pos[min(k, len(pos) - 1)])


def fit_thresholds(P, Y, label_ids, policy):
    th = []
    for j, lab in enumerate(label_ids):
        p, y = P[:, j], Y[:, j]
        if not (0 < y.sum() < len(y)):
            th.append(0.5)
        elif policy == "sens90" and lab in HIGH_RISK:
            th.append(sens_threshold(p, y, 0.90))
        else:
            th.append(best_f1_threshold(p, y))
    return np.array(th)


def cv_thresholds(P, Y, groups, label_ids, policy, n_splits=5, seed=0):
    """Pooled held-out binary predictions with thresholds fitted on the other folds."""
    pred = np.zeros_like(Y)
    for fit_idx, held_idx in group_folds(groups, n_splits, seed):
        th = fit_thresholds(P[fit_idx], Y[fit_idx], label_ids, policy)
        pred[held_idx] = (P[held_idx] >= th).astype(int)
    return pred


def binary_metrics(Y, pred, label_ids, tiers):
    rows = []
    for j, lab in enumerate(label_ids):
        y, q = Y[:, j], pred[:, j]
        tp, fp = int((q & y).sum()), int((q & (1 - y)).sum())
        fn, tn = int(((1 - q) & y).sum()), int(((1 - q) & (1 - y)).sum())
        sens = tp / (tp + fn) if tp + fn else np.nan
        spec = tn / (tn + fp) if tn + fp else np.nan
        prec = tp / (tp + fp) if tp + fp else np.nan
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
        rows.append({"label": lab, "tier": tiers[j], "n_pos": int(y.sum()), "tp": tp, "fp": fp,
                     "fn": fn, "sensitivity": sens, "specificity": spec, "precision": prec, "f1": f1})
    prim = [j for j, t in enumerate(tiers) if t == "primary"]
    macro_f1 = float(np.nanmean([rows[j]["f1"] for j in prim]))
    TP = sum(rows[j]["tp"] for j in prim)
    FP = sum(rows[j]["fp"] for j in prim)
    FN = sum(rows[j]["fn"] for j in prim)
    micro_f1 = 2 * TP / (2 * TP + FP + FN)
    return rows, {"macro_f1_primary": macro_f1, "micro_f1_primary": float(micro_f1)}


def calibration_table(Y, P_before, P_after, label_ids):
    rows = []
    for j, lab in enumerate(label_ids):
        y = Y[:, j]
        rows.append({"label": lab,
                     "ece_before": ece(y, P_before[:, j]), "ece_after": ece(y, P_after[:, j]),
                     "brier_before": brier_score_loss(y, P_before[:, j]),
                     "brier_after": brier_score_loss(y, P_after[:, j])})
    return rows


def fit_full(P, Y, label_ids, methods):
    """Calibrators (chosen methods) and thresholds fitted on the full fold 9, for later fold-10 use."""
    cals = [fit_calibrator(P[:, j], Y[:, j], methods[j]) for j in range(P.shape[1])]
    P_cal = np.stack([apply_calibrator(P[:, j], cals[j]) for j in range(P.shape[1])], axis=1)
    return {"labels": list(label_ids), "calibrators": cals,
            "thresholds": {pol: fit_thresholds(P_cal, Y, label_ids, pol).tolist()
                           for pol in ("max_f1", "sens90")},
            "high_risk_labels": HIGH_RISK}
