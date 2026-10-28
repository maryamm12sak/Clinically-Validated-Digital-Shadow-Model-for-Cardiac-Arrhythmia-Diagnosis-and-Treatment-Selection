"""
Ensembles of saved fold-9 predictions (ml/results/predictions/<run_id>.npz).

Seed handling: each base model's prediction is the average of its seed runs ("seed-averaged").
Equal-weight averaging is also reported seed-matched (seed s of every member, mean +- SD over seeds).

Fold 9 is reused for evaluation, so every method that fits something (weights, meta-learner) is
scored with 5-fold patient-grouped cross-validation inside fold 9: fit on 4/5, predict the held-out
1/5, pool the held-out predictions, then compute metrics on the pooled predictions.
Methods:
  mean      equal-weight average of member probabilities (nothing fitted)
  weighted  per-label soft voting, weight of a member = max(AUROC - 0.5, 0) on the fitting part
  stacking  per-label logistic regression on member logits
"""
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from .evaluate import evaluate

EPS = 1e-6


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def load_run(pred_dir, run_id):
    d = np.load(Path(pred_dir) / f"{run_id}.npz", allow_pickle=False)
    return d["scores"].astype(np.float64), d["labels"], d["ecg_id"]


def load_model(pred_dir, model_prefix, seeds):
    """Seed runs of one model -> (list of per-seed scores, labels, ecg_id); checks alignment."""
    runs = [load_run(pred_dir, f"{model_prefix}__seed{s}") for s in seeds]
    ids, labels = runs[0][2], runs[0][1]
    for _, y, e in runs[1:]:
        assert np.array_equal(e, ids) and np.array_equal(y, labels), "prediction files not aligned"
    return [r[0] for r in runs], labels, ids


def group_folds(groups, n_splits=5, seed=0):
    return list(GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(groups, groups=groups))


def _auroc_or_nan(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def fit_weights(members, Y):
    """Per-label member weights (L, M) from AUROC - 0.5 on the given data."""
    W = np.zeros((Y.shape[1], len(members)))
    for j in range(Y.shape[1]):
        for m, P in enumerate(members):
            a = _auroc_or_nan(Y[:, j], P[:, j])
            W[j, m] = max(a - 0.5, 0.0) if np.isfinite(a) else 1.0
        W[j] = W[j] / W[j].sum() if W[j].sum() > 0 else 1.0 / len(members)
    return W


def apply_weights(members, W):
    return np.stack([sum(W[j, m] * P[:, j] for m, P in enumerate(members))
                     for j in range(W.shape[0])], axis=1)


def fit_stacker(members, Y, C=1.0):
    models = []
    for j in range(Y.shape[1]):
        X = np.column_stack([logit(P[:, j]) for P in members])
        y = Y[:, j]
        if 0 < y.sum() < len(y):
            models.append(LogisticRegression(C=C, max_iter=1000).fit(X, y))
        else:
            models.append(None)
    return models


def apply_stacker(members, models):
    out = []
    for j, m in enumerate(models):
        X = np.column_stack([logit(P[:, j]) for P in members])
        out.append(m.predict_proba(X)[:, 1] if m is not None else np.full(len(X), 0.5))
    return np.stack(out, axis=1)


def cross_validated(members, Y, groups, method, n_splits=5, seed=0):
    """Pooled out-of-fold predictions for a fitted combination method."""
    P = np.zeros_like(members[0])
    for fit_idx, held_idx in group_folds(groups, n_splits, seed):
        fit_members = [M[fit_idx] for M in members]
        held_members = [M[held_idx] for M in members]
        if method == "weighted":
            P[held_idx] = apply_weights(held_members, fit_weights(fit_members, Y[fit_idx]))
        elif method == "stacking":
            P[held_idx] = apply_stacker(held_members, fit_stacker(fit_members, Y[fit_idx]))
        else:
            raise ValueError(method)
    return P


def summarize(Y, P, labels, groups, n_boot=1000):
    per_label, s = evaluate(Y, P, labels, groups, n_boot=n_boot, seed=0)
    return per_label, {k: s.get(k, np.nan)
                       for k in ("macro_auroc_primary", "macro_auprc_primary", "ci_low", "ci_high")}
