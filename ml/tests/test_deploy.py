"""Contract and parity tests for the deploy package (ml/deploy)."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pytest

ML_DIR = Path(__file__).resolve().parents[1]
DEPLOY = ML_DIR / "deploy"
REPO = ML_DIR.parent
if not (DEPLOY / "manifest.json").exists():
    pytest.skip("deploy package not built (run export_deploy.py)", allow_module_level=True)
sys.path.insert(0, str(DEPLOY))

import ecg_model  # noqa: E402
import features as dfeat  # noqa: E402
import preprocess as dpp  # noqa: E402
from src import feature_extraction as sfeat  # noqa: E402
from src import preprocessing as spp  # noqa: E402


def _ecg(seconds, fs, hr=70, seed=0):
    x = nk.ecg_simulate(duration=seconds, sampling_rate=fs, heart_rate=hr, random_state=seed)
    scale = np.linspace(0.5, 1.5, 12)
    return np.asarray(x)[:, None] * scale[None, :]


def _backend_label_ids():
    spec = importlib.util.spec_from_file_location("backend_labels", REPO / "labels.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.LABEL_IDS


def test_label_order_matches_backend():
    assert ecg_model.get_model().label_ids == _backend_label_ids()


@pytest.mark.parametrize("seconds,fs", [(5, 500), (10, 100), (60, 500), (10, 250)])
def test_predict_contract(seconds, fs):
    probs = ecg_model.predict(_ecg(seconds, fs), fs)
    assert list(probs) == _backend_label_ids()
    assert all(0.0 <= p <= 1.0 for p in probs.values())


def test_predict_detailed_fields():
    d = ecg_model.predict_detailed(_ecg(10, 500), 500)
    assert d["mode"] == "resnet_only" and d["components"] == ["resnet1d_x3"]
    v = d["labels"]["AFLT"]
    assert v["exploratory"] and "expert review" in v["note"]
    for key in ("raw_probability", "calibrated_probability", "threshold_max_f1",
                "threshold_high_sensitivity", "detected", "detected_high_sensitivity"):
        assert key in d["labels"]["AFIB"]
    assert not d["labels"]["AFIB"]["exploratory"]


def test_ensemble_low_fs_warning():
    d = ecg_model.predict_detailed(_ecg(10, 250), 250, mode="ensemble")
    assert d["components"] == ["resnet1d_x3", "lightgbm_x3"]
    assert any("upsampled" in w for w in d["warnings"])
    assert d["labels"]["NSR"]["lightgbm_probability"] is not None


@pytest.mark.parametrize("seconds", [4, 61])
def test_duration_limits(seconds):
    with pytest.raises(ValueError):
        ecg_model.predict(_ecg(seconds, 100), 100)


def test_bad_shape_and_nan():
    with pytest.raises(ValueError):
        ecg_model.predict(np.zeros((1000, 11)), 100)
    x = _ecg(10, 100)
    x[5, 3] = np.nan
    with pytest.raises(ValueError):
        ecg_model.predict(x, 100)


def test_default_mode_does_not_import_ensemble_dependencies():
    code = ("import sys, numpy as np; sys.path.insert(0, r'%s'); import ecg_model; "
            "ecg_model.predict(np.random.default_rng(0).normal(size=(1000, 12)) * 0.1, 100); "
            "print('lightgbm' in sys.modules, 'neurokit2' in sys.modules, 'pywt' in sys.modules)") % DEPLOY
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False", "False"]


def test_vendored_preprocessing_identical():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5000, 12))
    for fs_in, fs_out in ((500, 100), (250, 100), (100, 500), (100, 100)):
        assert np.array_equal(dpp.preprocess(x, fs_in, fs_out), spp.preprocess(x, fs_in, fs_out))
    names = [n.lower() for n in dpp.LEAD_ORDER][::-1]
    assert np.array_equal(dpp.reorder_leads(x[:, ::-1], names), spp.reorder_leads(x[:, ::-1], names))
    for n in (1000, 1030, 6000):
        a, sa = dpp.sliding_windows(x[:n].astype(np.float32), 250)
        b, sb = spp.sliding_windows(x[:n].astype(np.float32), 250)
        assert sa == sb and np.array_equal(a, b)


def test_vendored_features_identical():
    x = _ecg(10, 500, hr=85)
    a, b = dfeat.extract_features(x, 500), sfeat.extract_features(x, 500)
    assert list(a) == list(b)
    va, vb = np.array(list(a.values()), float), np.array(list(b.values()), float)
    assert np.array_equal(va, vb, equal_nan=True)


def test_manifest_hashes_match_files():
    import hashlib
    import json
    m = json.loads((DEPLOY / "manifest.json").read_text(encoding="utf-8"))
    for rel, h in m["sha256"].items():
        assert hashlib.sha256((DEPLOY / rel).read_bytes()).hexdigest() == h, rel
