import numpy as np
import pandas as pd

from src.labels import (build_label_matrix, count_table, exploratory_indices, label_ids,
                        load_label_config, primary_indices)

# Expected record counts (folds 1-8, fold 9, fold 10) from the PTB-XL audit.
AUDIT_COUNTS = {
    "NSR": (13404, 1670, 1674),
    "AFIB": (1211, 151, 152),
    "AVB1": (634, 80, 79),
    "LBBB": (428, 54, 54),
    "PVC": (915, 114, 114),
    "PAC": (318, 40, 40),
}


def test_label_order_matches_product():
    assert label_ids() == ["NSR", "SBRAD", "STACH", "AFIB", "AFLT", "SVT", "AVB1", "AVB_HIGH",
                           "LBBB", "RBBB", "PVC", "PAC"]


def test_tiers():
    ids = label_ids()
    assert [ids[j] for j in exploratory_indices()] == ["AFLT", "SVT", "AVB_HIGH"]
    assert len(primary_indices()) == 9
    assert not set(primary_indices()) & set(exploratory_indices())


def test_likelihood_zero_counts_as_present():
    Y = build_label_matrix([{"SR": 0.0}, {"AFIB": 0.0, "NORM": 100.0}, {"NORM": 100.0}])
    ids = label_ids()
    assert Y[0, ids.index("NSR")] == 1
    assert Y[1, ids.index("AFIB")] == 1
    assert Y[2].sum() == 0


def test_merged_groups_are_or_without_double_counting():
    ids = label_ids()
    Y = build_label_matrix([{"SVTAC": 0.0, "PSVT": 0.0}, {"PSVT": 0.0}, {"2AVB": 100.0, "3AVB": 50.0}])
    assert Y[:, ids.index("SVT")].tolist() == [1, 1, 0]
    assert Y[:, ids.index("AVB_HIGH")].tolist() == [0, 0, 1]
    assert Y.max() == 1


def test_counts_match_audit(dataset, config):
    table = count_table(dataset.db, config)
    for label, expected in AUDIT_COUNTS.items():
        assert tuple(table.loc[label, ["train", "val", "test"]]) == expected, label


def test_merged_group_counts_equal_union(dataset):
    codes = dataset.db["scp_codes"]
    Y = build_label_matrix(codes)
    ids = label_ids()
    for label, members in [("SVT", ["SVTAC", "PSVT"]), ("AVB_HIGH", ["2AVB", "3AVB"])]:
        union = codes.apply(lambda d: any(c in d for c in members)).sum()
        assert Y[:, ids.index(label)].sum() == union
        assert union <= sum(codes.apply(lambda d, c=c: c in d).sum() for c in members)


def test_matrix_shape(dataset):
    Y = build_label_matrix(dataset.db["scp_codes"], load_label_config())
    assert Y.shape == (len(dataset.db), 12)
    assert Y.dtype == np.uint8
    assert isinstance(count_table(dataset.db, {"splits": {"train": [1], "val": [9], "test": [10]}}),
                      pd.DataFrame)
