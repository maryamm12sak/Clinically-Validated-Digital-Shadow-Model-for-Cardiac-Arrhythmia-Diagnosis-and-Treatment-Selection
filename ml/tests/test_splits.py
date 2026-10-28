import copy

import numpy as np
import pytest

from src.data_loader import LockedSplitError, assert_no_patient_leakage


def test_no_patient_leakage(dataset, config):
    assert_no_patient_leakage(dataset.db, config)


def test_leakage_is_detected(dataset, config):
    db = dataset.db.copy()
    first_test = db.index[db["strat_fold"] == 10][0]
    first_train = db.index[db["strat_fold"] == 1][0]
    db.loc[first_test, "patient_id"] = db.loc[first_train, "patient_id"]
    with pytest.raises(AssertionError):
        assert_no_patient_leakage(db, config)


def test_split_sizes(dataset):
    assert len(dataset.get_split("train", load_signals=False).meta) == 17418
    assert len(dataset.get_split("val", load_signals=False).meta) == 2183
    assert len(dataset.get_split("test", allow_test=True, load_signals=False).meta) == 2198


def test_test_split_locked(dataset):
    with pytest.raises(LockedSplitError):
        dataset.get_split("test", load_signals=False)


def test_dev_subset(dataset, config):
    cfg = copy.deepcopy(config)
    cfg["dev_subset"] = {"enabled": True, "fraction": 0.1, "seed": 0}
    old = dataset.config
    try:
        dataset.config = cfg
        assert len(dataset.get_split("val", load_signals=False).meta) == round(2183 * 0.1)
    finally:
        dataset.config = old


def test_signal_cache_and_validation_report(dataset):
    X = dataset.signals()
    assert X.shape == (21799, 1000, 12)
    assert X.dtype == np.float32
    rep = dataset.validation_report()
    assert len(rep) == 21799
    assert (rep["reasons"].str.contains("FLAT_LEAD")).sum() == 1
    assert rep.loc[12722, "flat_leads"] == "V5"
    assert (rep["reasons"].str.contains("EXTREME_AMPLITUDE")).sum() == 25
    assert not rep["reasons"].str.contains("HAS_NAN|HAS_INF|BAD_SHAPE|READ_ERROR").any()


def test_val_split_signals_aligned(dataset):
    split = dataset.get_split("val")
    assert split.X.shape == (2183, 1000, 12)
    pos = dataset.db.index.get_loc(split.meta.index[5])
    assert np.array_equal(split.X[5], dataset.signals()[pos])
