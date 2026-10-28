"""
Threshold-free evaluation: per-label AUROC / AUPRC and macro-AUROC over the PRIMARY labels.

A metric is undefined (NaN, flagged) when a label has no positives or no negatives in the split;
such labels are excluded from the macro average and listed in `excluded_from_macro`.
Exploratory labels are reported per class but never enter the macro average.
Confidence intervals use a patient-level bootstrap: patients are resampled with replacement and
all of a patient's records come along.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def per_label_metrics(Y, P, labels):
    """Y, P: (N, L). labels: label config list. Returns DataFrame indexed by label id."""
    rows = []
    for j, lab in enumerate(labels):
        y, p = Y[:, j], P[:, j]
        n_pos = int(y.sum())
        defined = 0 < n_pos < len(y)
        rows.append({
            "label": lab["id"],
            "tier": lab["tier"],
            "n": len(y),
            "n_pos": n_pos,
            "prevalence": n_pos / len(y) if len(y) else np.nan,
            "auroc": roc_auc_score(y, p) if defined else np.nan,
            "auprc": average_precision_score(y, p) if defined else np.nan,
            "undefined": not defined,
        })
    return pd.DataFrame(rows).set_index("label")


def macro_summary(per_label):
    """Macro AUROC / AUPRC over primary labels whose metrics are defined."""
    prim = per_label[per_label["tier"] == "primary"]
    used = prim[~prim["undefined"]]
    return {
        "macro_auroc_primary": float(used["auroc"].mean()) if len(used) else np.nan,
        "macro_auprc_primary": float(used["auprc"].mean()) if len(used) else np.nan,
        "n_labels_in_macro": int(len(used)),
        "excluded_from_macro": list(prim.index[prim["undefined"]]),
    }


def bootstrap_macro_auroc(Y, P, labels, patient_ids, n_boot=1000, seed=0, alpha=0.05):
    """Patient-level bootstrap CI for macro-AUROC over primary labels. Returns (low, high)."""
    prim = [j for j, lab in enumerate(labels) if lab["tier"] == "primary"
            and 0 < Y[:, j].sum() < len(Y)]
    if not prim:
        return np.nan, np.nan
    patient_ids = np.asarray(patient_ids)
    uniq, inverse = np.unique(patient_ids, return_inverse=True)
    rows_of = [np.flatnonzero(inverse == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_of[k] for k in pick])
        aucs = []
        for j in prim:
            y = Y[idx, j]
            if 0 < y.sum() < len(y):
                aucs.append(roc_auc_score(y, P[idx, j]))
        if aucs:
            stats.append(np.mean(aucs))
    low, high = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def evaluate(Y, P, labels, patient_ids=None, n_boot=1000, seed=0):
    per_label = per_label_metrics(Y, P, labels)
    summary = macro_summary(per_label)
    if patient_ids is not None and n_boot:
        summary["ci_low"], summary["ci_high"] = bootstrap_macro_auroc(
            Y, P, labels, patient_ids, n_boot, seed)
    return per_label, summary
