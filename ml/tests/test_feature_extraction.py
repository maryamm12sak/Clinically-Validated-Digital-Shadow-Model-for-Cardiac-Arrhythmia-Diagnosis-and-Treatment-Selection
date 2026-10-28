import neurokit2 as nk
import numpy as np
import pytest

from src.feature_extraction import ERROR_COLUMNS, all_columns, extract_features, feature_names


def _simulated_12_lead(hr, seconds=12, fs=500, seed=0):
    ecg = nk.ecg_simulate(duration=seconds, sampling_rate=fs, heart_rate=hr, random_state=seed)
    return np.tile(np.asarray(ecg)[:, None], (1, 12))


@pytest.mark.parametrize("hr", [60, 120])
def test_heart_rate_on_simulated_ecg(hr):
    feats = extract_features(_simulated_12_lead(hr), fs=500)
    assert abs(feats["rr_hr_bpm"] - hr) <= 5
    assert feats["err_rpeaks"] == 0
    assert set(feats) == set(all_columns())


def test_flat_signal_gives_nans_and_flag():
    feats = extract_features(np.zeros((6000, 12)), fs=500)
    assert feats["err_rpeaks"] == 1 and feats["err_any"] == 1
    assert np.isnan(feats["rr_hr_bpm"]) and np.isnan(feats["int_qrs_ms"])
    assert set(feats) == set(all_columns())


def test_nan_input_does_not_raise():
    sig = _simulated_12_lead(80)
    sig[100:200, 3] = np.nan
    feats = extract_features(sig, fs=500)
    assert feats["err_spectral"] == 1
    assert set(feats) == set(all_columns())


def test_feature_names_stable_and_unique():
    names = feature_names()
    assert len(names) == len(set(names))
    assert names == feature_names()
    assert not set(names) & set(ERROR_COLUMNS)
