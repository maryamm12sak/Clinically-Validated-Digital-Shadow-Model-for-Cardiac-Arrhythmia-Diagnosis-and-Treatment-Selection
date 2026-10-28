"""
Hand-crafted features for the classical baselines (logistic regression, LightGBM).

Input: one raw 12-lead record (T, 12) in mV, leads in LEAD_ORDER, normally at 500 Hz.
No fitting or scaling happens here; that belongs in the sklearn Pipeline.
Any failure inside a record gives NaN features and a raised error flag, never an exception.

Feature groups and names (stable; see feature_names()):
  rr_*        heart rate and RR-interval statistics from R peaks (lead II, fallback V5 then I)
                rr_hr_bpm, rr_mean_ms, rr_sd_ms, rr_cv, rr_min_ms, rr_max_ms, rr_rmssd_ms,
                rr_pnn50, rr_sd_irregularity, rr_premature_count, rr_sampen
  int_*       delineation intervals (NeuroKit2 DWT), medians over beats, NaN where it fails
                int_pr_ms, int_qrs_ms, int_qt_ms, int_qtc_ms (Bazett), int_p_present_frac
  amp_<lead>_{r,s,t}       median R / S / T amplitude relative to the lead median, mV
  psd_<lead>_<band>        Welch band power, mV^2, bands 0.5-5, 5-15, 15-40, 40-100 Hz
  dwt_<lead>_<band>_{energy,entropy}   db4 level-5 sub-bands A5, D5..D1
Bookkeeping columns: meta_rpeak_lead (index into LEAD_ORDER, -1 if none), meta_n_beats,
err_rpeaks, err_intervals, err_amplitude, err_spectral, err_any (0/1 flags).
"""
import time
import warnings
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd
import pywt
import wfdb
from joblib import Parallel, delayed
from scipy.signal import welch

from .preprocessing import LEAD_ORDER, reorder_leads

RR_FEATURES = ["rr_hr_bpm", "rr_mean_ms", "rr_sd_ms", "rr_cv", "rr_min_ms", "rr_max_ms",
               "rr_rmssd_ms", "rr_pnn50", "rr_sd_irregularity", "rr_premature_count", "rr_sampen"]
INTERVAL_FEATURES = ["int_pr_ms", "int_qrs_ms", "int_qt_ms", "int_qtc_ms", "int_p_present_frac"]
PSD_BANDS = [("0.5-5", 0.5, 5.0), ("5-15", 5.0, 15.0), ("15-40", 15.0, 40.0), ("40-100", 40.0, 100.0)]
DWT_WAVELET, DWT_LEVEL = "db4", 5
DWT_BANDS = ["A5", "D5", "D4", "D3", "D2", "D1"]
META_COLUMNS = ["meta_rpeak_lead", "meta_n_beats"]
ERROR_COLUMNS = ["err_rpeaks", "err_intervals", "err_amplitude", "err_spectral", "err_any"]
RPEAK_LEADS = ["II", "V5", "I"]


def feature_names():
    names = list(RR_FEATURES) + list(INTERVAL_FEATURES)
    names += [f"amp_{lead}_{w}" for lead in LEAD_ORDER for w in ("r", "s", "t")]
    names += [f"psd_{lead}_{b}" for lead in LEAD_ORDER for b, _, _ in PSD_BANDS]
    names += [f"dwt_{lead}_{b}_{k}" for lead in LEAD_ORDER for b in DWT_BANDS
              for k in ("energy", "entropy")]
    return names


def all_columns():
    return feature_names() + META_COLUMNS + ERROR_COLUMNS


def detect_r_peaks(sig, fs, leads=RPEAK_LEADS):
    """R peaks on the first lead in `leads` where detection looks plausible. Returns (peaks, lead index)."""
    for lead in leads:
        j = LEAD_ORDER.index(lead)
        x = sig[:, j]
        if not np.isfinite(x).all() or np.ptp(x) == 0:
            continue
        try:
            cleaned = nk.ecg_clean(x, sampling_rate=fs)
            _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
            peaks = np.asarray(info["ECG_R_Peaks"], dtype=int)
        except Exception:
            continue
        duration_s = len(x) / fs
        if len(peaks) >= 3 and 20 <= 60 * len(peaks) / duration_s <= 300:
            return peaks, j
    return None, -1


def rr_features(peaks, fs):
    rr = np.diff(peaks) / fs * 1000.0
    d = np.diff(rr)
    med = np.median(rr)
    out = {
        "rr_hr_bpm": 60000.0 / rr.mean(),
        "rr_mean_ms": rr.mean(),
        "rr_sd_ms": rr.std(ddof=1) if len(rr) > 1 else np.nan,
        "rr_min_ms": rr.min(),
        "rr_max_ms": rr.max(),
        "rr_rmssd_ms": np.sqrt(np.mean(d ** 2)) if len(d) else np.nan,
        "rr_pnn50": np.mean(np.abs(d) > 50) if len(d) else np.nan,
        "rr_sd_irregularity": np.std(np.abs(d)) / rr.mean() if len(d) else np.nan,
        "rr_premature_count": float(np.sum(rr < 0.8 * med)),
    }
    out["rr_cv"] = out["rr_sd_ms"] / out["rr_mean_ms"]
    try:
        out["rr_sampen"] = float(nk.entropy_sample(rr)[0]) if len(rr) >= 5 else np.nan
    except Exception:
        out["rr_sampen"] = np.nan
    return out


def _wave(waves, key):
    return np.asarray(waves.get(key, []), dtype=float)


def interval_features(sig, fs, peaks, lead_idx):
    cleaned = nk.ecg_clean(sig[:, lead_idx], sampling_rate=fs)
    _, waves = nk.ecg_delineate(cleaned, peaks, sampling_rate=fs, method="dwt")
    n = len(peaks)

    def per_beat(key):
        v = _wave(waves, key)
        return v[:n] if len(v) >= n else np.full(n, np.nan)

    p_on, p_peak = per_beat("ECG_P_Onsets"), per_beat("ECG_P_Peaks")
    r_on, r_off, t_off = per_beat("ECG_R_Onsets"), per_beat("ECG_R_Offsets"), per_beat("ECG_T_Offsets")
    to_ms = 1000.0 / fs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pr = np.nanmedian((r_on - p_on) * to_ms)
        qrs = np.nanmedian((r_off - r_on) * to_ms)
        qt = np.nanmedian((t_off - r_on) * to_ms)
    rr_s = np.median(np.diff(peaks)) / fs
    return {
        "int_pr_ms": pr,
        "int_qrs_ms": qrs,
        "int_qt_ms": qt,
        "int_qtc_ms": qt / np.sqrt(rr_s) if np.isfinite(qt) else np.nan,
        "int_p_present_frac": float(np.mean(np.isfinite(p_peak))),
    }


def amplitude_features(sig, fs, peaks):
    out = {}
    w_r, w_s = int(0.05 * fs), int(0.08 * fs)
    t_lo, t_hi = int(0.15 * fs), int(0.40 * fs)
    T = len(sig)
    for j, lead in enumerate(LEAD_ORDER):
        x = sig[:, j] - np.median(sig[:, j])
        r, s, t = [], [], []
        for p in peaks:
            if p - w_r < 0 or p + t_hi > T:
                continue
            r.append(x[p - w_r:p + w_r].max())
            s.append(x[p:p + w_s].min())
            seg = x[p + t_lo:p + t_hi]
            t.append(seg[np.argmax(np.abs(seg))])
        out[f"amp_{lead}_r"] = np.median(r) if r else np.nan
        out[f"amp_{lead}_s"] = np.median(s) if s else np.nan
        out[f"amp_{lead}_t"] = np.median(t) if t else np.nan
    return out


def spectral_features(sig, fs):
    out = {}
    nperseg = min(len(sig), int(2 * fs))
    freqs, pxx = welch(sig, fs=fs, nperseg=nperseg, axis=0)
    df = freqs[1] - freqs[0]
    for j, lead in enumerate(LEAD_ORDER):
        for name, lo, hi in PSD_BANDS:
            m = (freqs >= lo) & (freqs < min(hi, fs / 2))
            out[f"psd_{lead}_{name}"] = float(pxx[m, j].sum() * df) if m.any() else np.nan
        coeffs = pywt.wavedec(sig[:, j], DWT_WAVELET, level=DWT_LEVEL)
        for band, c in zip(DWT_BANDS, coeffs):
            e = c ** 2
            total = e.sum()
            out[f"dwt_{lead}_{band}_energy"] = float(total)
            if total > 0:
                p = e[e > 0] / total
                out[f"dwt_{lead}_{band}_entropy"] = float(-(p * np.log2(p)).sum())
            else:
                out[f"dwt_{lead}_{band}_entropy"] = np.nan
    return out


def extract_features(sig, fs=500):
    """Features for one record (T, 12) in mV. Always returns a dict with all_columns() keys."""
    feats = dict.fromkeys(all_columns(), np.nan)
    flags = dict.fromkeys(ERROR_COLUMNS, 0)
    sig = np.asarray(sig, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            peaks, lead_idx = detect_r_peaks(sig, fs)
        except Exception:
            peaks, lead_idx = None, -1
        feats["meta_rpeak_lead"] = lead_idx
        feats["meta_n_beats"] = 0 if peaks is None else len(peaks)

        if peaks is None:
            flags["err_rpeaks"] = flags["err_intervals"] = flags["err_amplitude"] = 1
        else:
            try:
                feats.update(rr_features(peaks, fs))
            except Exception:
                flags["err_rpeaks"] = 1
            try:
                feats.update(interval_features(sig, fs, peaks, lead_idx))
            except Exception:
                flags["err_intervals"] = 1
            try:
                feats.update(amplitude_features(sig, fs, peaks))
            except Exception:
                flags["err_amplitude"] = 1
        try:
            if not np.isfinite(sig).all():
                raise ValueError("non-finite samples")
            feats.update(spectral_features(sig, fs))
        except Exception:
            flags["err_spectral"] = 1
    flags["err_any"] = int(any(flags.values()))
    # Undefined values (e.g. sample entropy with no template matches on ~10 RR intervals) -> NaN
    for name in feature_names():
        v = feats[name]
        if v is not None and not np.isfinite(v):
            feats[name] = np.nan
    feats.update(flags)
    return feats


def _extract_record(path, fs):
    try:
        sig, meta = wfdb.rdsamp(path)
        sig = reorder_leads(sig, meta["sig_name"])
    except Exception:
        feats = dict.fromkeys(all_columns(), np.nan)
        feats.update(dict.fromkeys(ERROR_COLUMNS, 1))
        return feats
    return extract_features(sig, fs)


def extract_ptbxl_features(dataset, ecg_ids=None, n_jobs=-1, dev_fraction=None, seed=0,
                           chunk_size=2000, progress=True):
    """
    Features for PTB-XL records at 500 Hz, read straight from the WFDB files.
    dataset: data_loader.PTBXL. ecg_ids: subset (default all). dev_fraction: random fraction for
    quick development runs. Returns a DataFrame indexed by ecg_id with all_columns().
    """
    ids = np.asarray(dataset.db.index if ecg_ids is None else ecg_ids)
    if dev_fraction:
        rng = np.random.default_rng(seed)
        ids = np.sort(rng.choice(ids, size=max(1, int(round(len(ids) * dev_fraction))), replace=False))
    paths = [str(dataset.root / dataset.db.at[i, "filename_hr"]) for i in ids]
    rows = []
    with Parallel(n_jobs=n_jobs) as parallel:
        for start in range(0, len(paths), chunk_size):
            rows += parallel(delayed(_extract_record)(p, 500) for p in paths[start:start + chunk_size])
            if progress:
                print(f"  features: {min(start + chunk_size, len(paths))}/{len(paths)}", flush=True)
    return pd.DataFrame(rows, index=pd.Index(ids, name="ecg_id"), columns=all_columns())


def load_or_build_features(dataset, path, force=False, n_jobs=-1, dev_fraction=None):
    """
    Read the feature table from `path` (CSV, ecg_id index) or extract and save it.
    Returns (features DataFrame, info dict with runtime_s, n_records, failure rates, mostly-NaN features).
    """
    path = Path(path)
    info = {"path": str(path), "built": False}
    if path.exists() and not force:
        feats = pd.read_csv(path, index_col="ecg_id")
        if np.isinf(feats[feature_names()].to_numpy(dtype=float)).any():
            raise ValueError(f"{path} contains infinite values; rebuild it with force=True")
    else:
        t0 = time.time()
        feats = extract_ptbxl_features(dataset, n_jobs=n_jobs, dev_fraction=dev_fraction)
        info["runtime_s"] = round(time.time() - t0, 1)
        info["built"] = True
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".partial.csv")
        feats.to_csv(tmp)
        tmp.replace(path)
    info["n_records"] = len(feats)
    info["failure_rate"] = {c: round(float(feats[c].mean()), 4) for c in ERROR_COLUMNS}
    nan_frac = feats[feature_names()].isna().mean()
    info["mostly_nan_features"] = {k: round(float(v), 3) for k, v in nan_frac[nan_frac > 0.5].items()}
    info["max_nan_fraction_by_group"] = {
        g: round(float(nan_frac[[n for n in feature_names() if n.startswith(g)]].max()), 3)
        for g in ("rr_", "int_", "amp_", "psd_", "dwt_")}
    return feats, info
