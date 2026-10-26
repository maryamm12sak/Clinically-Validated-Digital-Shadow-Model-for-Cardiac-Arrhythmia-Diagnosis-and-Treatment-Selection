"""
Heart rate and ECG intervals (PR, QRS, QT, QTc) from a 12-lead ECG.

How it works:
1. QRS start/end are found from ALL 12 leads combined (the "global" QRS):
   where the signal changes fastest across all leads at once. This replaced NeuroKit2's
   single-lead method, which over-measured QRS by ~70 ms.
2. PR uses a MEDIAN BEAT: all beats are lined up on their R peak and the median taken at each
   point, which cancels noise (ECG machines do the same). On that clean beat, the P wave is the
   group of slope bumps just before the QRS; PR = QRS start - start of that group.
   This replaced per-beat P detection, which mistook baseline noise for P waves (e.g. PR 270
   on a sinus bradycardia whose true PR is ~200).
3. T-wave end comes from NeuroKit2 (dwt) on leads I, II, V5, V6; QT = T end - QRS start.
4. QTc = QT / sqrt(RR)  (Bazett).
Values are validated against known PTB-XL diagnoses (see the project notes).
"""
import numpy as np
import neurokit2 as nk

PLAUSIBLE = {"pr": (80, 400), "qrs": (40, 200), "qt": (200, 700)}
T_LEADS = [0, 1, 10, 11]   # I, II, V5, V6 for T waves
QRS_THRESHOLD = 0.15       # fraction of peak slope that counts as "inside the QRS"
P_THRESHOLD = 0.3          # fraction of P-region slope range that counts as "inside the P wave"


def _clean(x):
    """Rounded int, or None if it couldn't be measured."""
    return None if x is None or np.isnan(x) else int(round(x))


def _spatial_slope(sig):
    """How fast the signal is changing, combined over all leads."""
    return np.sqrt((np.diff(sig, axis=0) ** 2).sum(axis=1))


def _qrs_bounds(slope, centre, fs):
    """QRS onset/offset around one R peak (sample indices), or None at the recording edge."""
    win = int(0.12 * fs)
    if centre - win < 0 or centre + win >= len(slope):
        return None
    seg = slope[centre - win:centre + win]
    above = np.where(seg > QRS_THRESHOLD * seg.max())[0]
    return centre - win + above.min(), centre - win + above.max()


def _global_qrs(cleaned, fs, r_peaks):
    slope = _spatial_slope(cleaned)
    k = max(1, int(0.01 * fs))
    slope = np.convolve(slope, np.ones(k) / k, mode="same")
    bounds = [b for b in (_qrs_bounds(slope, p, fs) for p in r_peaks) if b]
    return np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])


def _median_beat(cleaned, fs, r_peaks, rr_sec):
    pre, post = int(min(0.45, 0.6 * rr_sec) * fs), int(0.6 * fs)
    segs = [cleaned[p - pre:p + post] for p in r_peaks if p - pre >= 0 and p + post <= len(cleaned)]
    if len(segs) < 3:
        return None, None
    return np.median(np.stack(segs), axis=0), pre


def _pr_from_median_beat(med, r_idx, fs, rr_sec):
    """PR (ms) on the median beat, or nan if no P wave can be found."""
    raw = _spatial_slope(med)
    q = _qrs_bounds(raw, r_idx, fs)
    if q is None:
        return np.nan
    q_on = q[0]

    k = max(1, int(0.03 * fs))
    slope = np.convolve(raw, np.ones(k) / k, mode="same")
    lo = max(0, q_on - int(min(0.32, 0.5 * rr_sec) * fs))
    hi = q_on - int(0.04 * fs)
    s = slope[lo:hi]
    if len(s) < 3:
        return np.nan
    floor, peak = np.percentile(s, 10), s.max()
    idx = np.where(s > floor + P_THRESHOLD * (peak - floor))[0]
    if len(idx) == 0:
        return np.nan

    # contiguous slope bumps; walk back from the QRS, keeping bumps less than 50 ms apart (one P wave)
    starts = [idx[0]] + [idx[i] for i in range(1, len(idx)) if idx[i] != idx[i - 1] + 1]
    ends = [idx[i] for i in range(len(idx) - 1) if idx[i + 1] != idx[i] + 1] + [idx[-1]]
    gap = int(0.05 * fs)
    j = len(starts) - 1
    if ends[j] >= len(s) - 2 and j > 0 and starts[j] - ends[j - 1] > gap:
        j -= 1                                    # skip the QRS upstroke leaking into the window
    while j > 0 and starts[j] - ends[j - 1] <= gap:
        j -= 1
    return (q_on - (lo + starts[j])) * 1000 / fs


def compute_features(ecg, fs):
    """ecg: array of shape (samples, 12). Returns heart rate and intervals in ms."""
    cleaned = np.column_stack([nk.ecg_clean(ecg[:, i], sampling_rate=fs) for i in range(ecg.shape[1])])
    ms = 1000 / fs

    # Heart rate from lead II
    _, rp = nk.ecg_peaks(cleaned[:, 1], sampling_rate=fs)
    r_peaks = rp["ECG_R_Peaks"]
    rr_sec = np.mean(np.diff(r_peaks)) / fs
    hr = 60 / rr_sec

    # QRS from all leads, beat by beat
    q_on, q_off = _global_qrs(cleaned, fs, r_peaks)
    qrs_vals = (q_off - q_on) * ms

    # PR from the median beat
    med, r_idx = _median_beat(cleaned, fs, r_peaks, np.median(np.diff(r_peaks)) / fs)
    pr = _pr_from_median_beat(med, r_idx, fs, np.median(np.diff(r_peaks)) / fs) if med is not None else np.nan

    # QT: T end from several leads, matched to each beat's QRS start
    qt_vals = []
    for i in T_LEADS:
        try:
            _, lead_rp = nk.ecg_peaks(cleaned[:, i], sampling_rate=fs)
            _, w = nk.ecg_delineate(cleaned[:, i], lead_rp, sampling_rate=fs, method="dwt")
        except Exception:
            continue                                  # skip a lead that's too noisy to measure
        t_off = np.array(w.get("ECG_T_Offsets", []), dtype=float)
        t_off = t_off[~np.isnan(t_off)]
        for o in q_on:
            t = t_off[(t_off > o + 0.2 * fs) & (t_off < o + 0.65 * fs)]   # T end 200-650 ms after QRS
            if len(t):
                qt_vals.append((t.min() - o) * ms)

    def plausible_median(vals, key):
        lo, hi = PLAUSIBLE[key]
        vals = np.array(vals, dtype=float)
        vals = vals[(vals >= lo) & (vals <= hi)]
        return np.median(vals) if len(vals) else np.nan

    qrs, qt = plausible_median(qrs_vals, "qrs"), plausible_median(qt_vals, "qt")
    pr = pr if PLAUSIBLE["pr"][0] <= pr <= PLAUSIBLE["pr"][1] else np.nan
    qtc = qt / np.sqrt(rr_sec)

    return {"heart_rate": _clean(hr), "pr_ms": _clean(pr), "qrs_ms": _clean(qrs),
            "qt_ms": _clean(qt), "qtc_ms": _clean(qtc)}
