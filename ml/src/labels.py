"""
Multi-hot label matrix for the 12 product labels (ml/configs/labels.yaml).

A label is positive when any of its SCP code keys is present in scp_codes, whatever the
likelihood value. Merged groups (SVT, AVB_HIGH) are a logical OR, so a record carrying both
codes counts once.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ML_DIR = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ML_DIR / "configs" / "labels.yaml"


def load_label_config(path=None):
    with open(path or DEFAULT_LABELS, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    labels = cfg["labels"]
    for lab in labels:
        lab["scp_codes"] = [str(c) for c in lab["scp_codes"]]
    return labels


def label_ids(labels=None):
    return [lab["id"] for lab in (labels or load_label_config())]


def build_label_matrix(scp_codes, labels=None):
    """scp_codes: iterable of dicts (parsed scp_codes column). Returns uint8 array (N, n_labels)."""
    labels = labels or load_label_config()
    scp_codes = list(scp_codes)
    Y = np.zeros((len(scp_codes), len(labels)), dtype=np.uint8)
    for j, lab in enumerate(labels):
        codes = lab["scp_codes"]
        Y[:, j] = [any(c in d for c in codes) for d in scp_codes]
    return Y


def tier_indices(tier, labels=None):
    labels = labels or load_label_config()
    return [j for j, lab in enumerate(labels) if lab["tier"] == tier]


def primary_indices(labels=None):
    return tier_indices("primary", labels)


def exploratory_indices(labels=None):
    return tier_indices("exploratory", labels)


def count_table(db, config, labels=None):
    """Positive record counts per label for each split in config['splits']."""
    labels = labels or load_label_config()
    Y = build_label_matrix(db["scp_codes"], labels)
    cols = {}
    for name in ("train", "val", "test"):
        mask = db["strat_fold"].isin(config["splits"][name]).values
        cols[name] = Y[mask].sum(axis=0)
    table = pd.DataFrame(cols, index=label_ids(labels))
    table["total"] = Y.sum(axis=0)
    table["tier"] = [lab["tier"] for lab in labels]
    return table
