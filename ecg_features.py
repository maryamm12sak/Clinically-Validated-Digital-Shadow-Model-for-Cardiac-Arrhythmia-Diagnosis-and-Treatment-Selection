"""
Heart rate and ECG intervals (PR, QRS, QT, QTc) from a 12-lead ECG.

How it works:
1. QRS start/end are found from ALL 12 leads combined (the "global" QRS):
   we measure how fast the signal is changing across every lead at once, and the
   QRS is where that change is above 15% of its peak. This is far more stable
   than NeuroKit2's single-lead dwt boundaries, which over-measured QRS by ~70 ms.
2. P-wave start and T-wave end come from NeuroKit2 (dwt) on leads I, II, V5, V6.
3. PR = QRS start - P start;  QT = T end - QRS start;  QTc = QT / sqrt(RR) (Bazett).
4. Each interval is the median over all beats and leads, after removing
   physically impossible values.
"""
import numpy as np
import neurokit2 as nk

PLAUSIBLE = {"pr": (80, 400), "qrs": (40, 200), "qt": (200, 700)}
LEADS = [0, 1, 10, 11]   # I, II, V5, V6 for P and T waves
QRS_THRESHOLD = 0.15     # fraction of peak slope that counts as "inside the QRS"


def _clean(x):
    """Rounded int, or None if it couldn't be measured."""
    return None if x is None or np.isnan(x) else int(round(x))


def _global_qrs(cleaned, fs, r_peaks):
    """QRS onset/offset (sample indices) for each beat, using all leads together."""
    slope = np.sqrt((np.diff(cleaned, axis=0) ** 2).sum(axis=1))
    k = max(1, int(0.01 * fs))                        # 10 ms smoothing
    slope = np.convolve(slope, np.ones(k) / k, mode="same")

    win = int(0.12 * fs)                              # look 120 ms either side of each R peak
    onsets, offsets = [], []
    for p in r_peaks:
        if p - win < 0 or p + win >= len(slope):
            continue                                  # beat cut off at the edge of the recording
        seg = slope[p - win:p + win]
        above = np.where(seg > QRS_THRESHOLD * seg.max())[0]
        onsets.append(p - win + above.min())
        offsets.append(p - win + above.max())
    return np.array(onsets), np.array(offsets)


def compute_features(ecg, fs):
    """ecg: array of shape (samples, 12). Returns heart rate and intervals in ms."""
    cleaned = np.column_stack([nk.ecg_clean(ecg[:, i], sampling_rate=fs) for i in range(ecg.shape[1])])
    ms = 1000 / fs

    # Heart rate from lead II
    _, rp = nk.ecg_peaks(cleaned[:, 1], sampling_rate=fs)
    r_peaks = rp["ECG_R_Peaks"]
    rr_sec = np.mean(np.diff(r_peaks)) / fs
    hr = 60 / rr_sec

    # QRS from all leads
    q_on, q_off = _global_qrs(cleaned, fs, r_peaks)
    pooled = {"pr": [], "qrs": list((q_off - q_on) * ms), "qt": []}

    # P start and T end from several leads, matched to each beat's QRS start
    for i in LEADS:
        try:
            _, lead_rp = nk.ecg_peaks(cleaned[:, i], sampling_rate=fs)
            _, w = nk.ecg_delineate(cleaned[:, i], lead_rp, sampling_rate=fs, method="dwt")
        except Exception:
            continue                                  # skip a lead that's too noisy to measure
        p_on = np.array(w.get("ECG_P_Onsets", []), dtype=float)
        t_off = np.array(w.get("ECG_T_Offsets", []), dtype=float)
        p_on, t_off = p_on[~np.isnan(p_on)], t_off[~np.isnan(t_off)]

        for o in q_on:
            p = p_on[(p_on < o) & (p_on > o - 0.35 * fs)]          # P start in the 350 ms before QRS
            if len(p):
                pooled["pr"].append((o - p.max()) * ms)
            t = t_off[(t_off > o + 0.2 * fs) & (t_off < o + 0.65 * fs)]  # T end 200-650 ms after QRS
            if len(t):
                pooled["qt"].append((t.min() - o) * ms)

    med = {}
    for key, (lo, hi) in PLAUSIBLE.items():
        vals = np.array(pooled[key], dtype=float)
        vals = vals[(vals >= lo) & (vals <= hi)]
        med[key] = np.median(vals) if len(vals) else np.nan

    qtc = med["qt"] / np.sqrt(rr_sec)

    return {"heart_rate": _clean(hr), "pr_ms": _clean(med["pr"]), "qrs_ms": _clean(med["qrs"]),
            "qt_ms": _clean(med["qt"]), "qtc_ms": _clean(qtc)}
