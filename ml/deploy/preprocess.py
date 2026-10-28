"""
Standalone preprocessing for the deployed model (vendored copy; no imports from ml/src).

Mirrors ml/src/preprocessing.py function for function; ml/tests/test_deploy.py checks that both
give identical outputs. Dependencies: numpy, scipy. (Hand-crafted features for the ensemble's
LightGBM part are in features.py.)

ResNet1d path: (T, 12) mV -> lead order check -> resample to 100 Hz -> per-lead z-score with the
saved training statistics -> 2.5 s windows with 50% overlap covering the whole signal.
"""
import json
from math import gcd

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
_LEAD_KEYS = [n.upper() for n in LEAD_ORDER]


# ---- signal preprocessing (ml/src/preprocessing.py) --------------------------------------------

def _key(name):
    return name.strip().upper().replace(" ", "").replace("-", "")


def reorder_leads(sig, lead_names):
    """Return sig (T, n_leads) with columns in LEAD_ORDER. Lead names are matched case-insensitively."""
    if sig.ndim != 2:
        raise ValueError(f"expected (T, leads), got shape {sig.shape}")
    keys = [_key(n) for n in lead_names]
    if len(keys) != sig.shape[1]:
        raise ValueError(f"{len(keys)} lead names for {sig.shape[1]} signal columns")
    missing = [LEAD_ORDER[i] for i, k in enumerate(_LEAD_KEYS) if k not in keys]
    if missing:
        raise ValueError(f"missing leads: {missing}")
    return sig[:, [keys.index(k) for k in _LEAD_KEYS]]


def resample(sig, fs_in, fs_out):
    """Polyphase resampling along time (axis 0)."""
    if fs_in == fs_out:
        return sig
    g = gcd(int(fs_in), int(fs_out))
    return resample_poly(sig, int(fs_out) // g, int(fs_in) // g, axis=0)


def bandpass(sig, fs, low_hz=0.5, high_hz=40.0, order=4):
    """Zero-phase Butterworth bandpass along time."""
    high = min(high_hz, 0.45 * fs)
    sos = butter(order, [low_hz, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, sig, axis=0)


def preprocess(sig, fs, target_fs, lead_names=None, bandpass_cfg=None):
    """sig: (T, 12) in mV. Returns float32 (T', 12) at target_fs."""
    sig = np.asarray(sig, dtype=np.float64)
    if lead_names is not None:
        sig = reorder_leads(sig, lead_names)
    elif sig.shape[1] != 12:
        raise ValueError(f"expected 12 leads, got {sig.shape[1]}")
    sig = resample(sig, fs, target_fs)
    if bandpass_cfg and bandpass_cfg.get("enabled", False):
        sig = bandpass(sig, target_fs, bandpass_cfg.get("low_hz", 0.5),
                       bandpass_cfg.get("high_hz", 40.0), bandpass_cfg.get("order", 4))
    return sig.astype(np.float32)


class PerLeadNormalizer:
    """Per-lead z-score with statistics saved from the training folds (normalizer_seed*.json)."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    def transform(self, X):
        return ((np.asarray(X, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        if d["lead_order"] != LEAD_ORDER:
            raise ValueError("normalizer was saved with a different lead order")
        return cls(d["mean"], d["std"])


def _pad_to(sig, length):
    if len(sig) >= length:
        return sig
    return np.pad(sig, ((0, length - len(sig)), (0, 0)))


def window_starts(n, window, overlap=0.5):
    """Start indices covering every sample of a length-n signal; the last window ends at n."""
    if n <= window:
        return [0]
    step = max(1, int(round(window * (1 - overlap))))
    starts = list(range(0, n - window + 1, step))
    if starts[-1] + window < n:
        starts.append(n - window)
    return starts


def sliding_windows(sig, window, overlap=0.5):
    """Returns (windows (K, window, 12), starts). Covers the full signal including the tail."""
    sig = _pad_to(sig, window)
    starts = window_starts(len(sig), window, overlap)
    return np.stack([sig[s:s + window] for s in starts]), starts
