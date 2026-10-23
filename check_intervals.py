import wfdb
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt

rec = wfdb.rdrecord('00001_hr', pn_dir='ptb-xl/1.0.3/records500/00000')
fs = rec.fs
lead = nk.ecg_clean(rec.p_signal[:, 1], sampling_rate=fs)
_, rpeaks = nk.ecg_peaks(lead, sampling_rate=fs)
ms = 1000 / fs

def interval(waves, start, end):
    a = np.array(waves.get(start, []), dtype=float)
    b = np.array(waves.get(end, []), dtype=float)
    n = min(len(a), len(b))
    return None if n == 0 else round(float(np.nanmedian((b[:n] - a[:n]) * ms)))

for method in ["dwt", "cwt"]:
    _, waves = nk.ecg_delineate(lead, rpeaks, sampling_rate=fs, method=method)
    print(f"{method}: PR={interval(waves, 'ECG_P_Onsets', 'ECG_R_Onsets')}  "
          f"QRS={interval(waves, 'ECG_R_Onsets', 'ECG_R_Offsets')}  "
          f"QT={interval(waves, 'ECG_R_Onsets', 'ECG_T_Offsets')}")

# Plot where dwt places each point (first 3 seconds)
_, waves = nk.ecg_delineate(lead, rpeaks, sampling_rate=fs, method="dwt")
t = np.arange(len(lead)) / fs
plt.plot(t, lead, color="gray")
for key, color in [("ECG_P_Onsets", "green"), ("ECG_R_Onsets", "red"),
                   ("ECG_R_Offsets", "blue"), ("ECG_T_Offsets", "purple")]:
    idx = np.array(waves[key], dtype=float)
    idx = idx[~np.isnan(idx)].astype(int)
    plt.scatter(t[idx], lead[idx], color=color, label=key, zorder=3)
plt.xlim(0, 3); plt.legend(); plt.title("Where dwt places the points"); plt.show()