"""
Signal preprocessing shared by training and inference.

Both paths go through preprocess(): lead check/reorder -> resample -> optional bandpass.
Normalization is a separate fitted object (PerLeadNormalizer), fitted on the train split only.
Training then takes random_crop() windows; inference takes sliding_windows() over the whole
signal, including the tail. Inputs of any length from 5 to 60 s are handled.
"""
import json
from math import gcd

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
_LEAD_KEYS = [n.upper() for n in LEAD_ORDER]


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
    """
    The single preprocessing path for training and inference.
    sig: (T, 12) in mV. lead_names: if given, leads are checked and reordered to LEAD_ORDER.
    Returns float32 (T', 12) at target_fs.
    """
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
    """Per-lead z-score. fit() on the training split only; reuse the saved JSON at inference."""

    def __init__(self, mean=None, std=None, eps=1e-6):
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)
        self.eps = eps

    def fit(self, X):
        """X: (N, T, 12) or (T, 12)."""
        flat = np.asarray(X, dtype=np.float64).reshape(-1, np.shape(X)[-1])
        self.mean = flat.mean(axis=0).astype(np.float32)
        self.std = np.maximum(flat.std(axis=0), self.eps).astype(np.float32)
        return self

    def transform(self, X):
        if self.mean is None:
            raise RuntimeError("normalizer is not fitted")
        return ((np.asarray(X, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"lead_order": LEAD_ORDER, "mean": self.mean.tolist(),
                       "std": self.std.tolist(), "eps": self.eps}, fh, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        if d["lead_order"] != LEAD_ORDER:
            raise ValueError("normalizer was saved with a different lead order")
        return cls(d["mean"], d["std"], d["eps"])


def _pad_to(sig, length):
    if len(sig) >= length:
        return sig
    return np.pad(sig, ((0, length - len(sig)), (0, 0)))


def random_crop(sig, window, rng):
    """One random window of `window` samples from (T, 12); zero-padded if T < window."""
    sig = _pad_to(sig, window)
    start = int(rng.integers(0, len(sig) - window + 1))
    return sig[start:start + window]


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


def training_example(sig, fs, target_fs, window_s, rng, normalizer=None, lead_names=None,
                     bandpass_cfg=None):
    x = preprocess(sig, fs, target_fs, lead_names, bandpass_cfg)
    if normalizer is not None:
        x = normalizer.transform(x)
    return random_crop(x, int(round(window_s * target_fs)), rng)


def inference_windows(sig, fs, target_fs, window_s, normalizer=None, lead_names=None,
                      bandpass_cfg=None, overlap=0.5):
    x = preprocess(sig, fs, target_fs, lead_names, bandpass_cfg)
    if normalizer is not None:
        x = normalizer.transform(x)
    return sliding_windows(x, int(round(window_s * target_fs)), overlap)
