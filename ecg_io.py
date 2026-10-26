"""
Reading uploaded ECG files into a (samples, 12) numpy array.

Supported inputs:
- CSV: one column per lead, optional header row with lead names (I, II, ... V6).
       Sampling rate is taken from the `fs` argument, or assumed from a 10-second length.
- WFDB: a .hea file plus its .dat file, uploaded together (the PTB-XL / PhysioNet format).

XML (hospital ECG machines) will be added once we know the machine brand.
"""
import io
import os
import tempfile

import numpy as np
import wfdb

STANDARD_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
DISPLAY_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


class ECGReadError(ValueError):
    """Raised when an upload can't be turned into a 12-lead ECG. The message is shown to the user."""


def _norm(name):
    return name.strip().upper().replace(" ", "").replace("-", "")


def _to_standard_order(sig, names):
    """Put the leads in the standard order I, II, III, aVR, aVL, aVF, V1-V6."""
    if names:
        normed = [_norm(n) for n in names]
        if all(lead in normed for lead in STANDARD_LEADS):
            return sig[:, [normed.index(lead) for lead in STANDARD_LEADS]]
    if sig.shape[1] == 12:
        return sig          # no usable names: assume the columns are already in standard order
    raise ECGReadError(f"Expected 12 leads, found {sig.shape[1]}.")


def _read_csv(content, fs=None):
    text = content.decode("utf-8-sig", errors="replace")
    first_line = text.splitlines()[0] if text else ""
    has_header = any(c.isalpha() for c in first_line)
    names = [n.strip() for n in first_line.split(",")] if has_header else None
    try:
        sig = np.loadtxt(io.StringIO(text), delimiter=",", skiprows=1 if has_header else 0, ndmin=2)
    except Exception:
        raise ECGReadError("Could not read the CSV. It should contain only numbers, one column per lead.")
    sig = _to_standard_order(sig, names)
    fs = fs or round(len(sig) / 10)            # PTB-XL-style recordings are 10 seconds long
    return sig, fs


def _read_wfdb(files):
    hea = [f for f in files if f[0].lower().endswith(".hea")]
    if not hea:
        raise ECGReadError("WFDB upload needs the .hea file together with its .dat file.")
    with tempfile.TemporaryDirectory() as tmp:
        for name, content in files:
            with open(os.path.join(tmp, os.path.basename(name)), "wb") as fh:
                fh.write(content)
        record_path = os.path.join(tmp, os.path.splitext(os.path.basename(hea[0][0]))[0])
        try:
            rec = wfdb.rdrecord(record_path)
        except Exception as e:
            raise ECGReadError(f"Could not read the WFDB record ({e}). Upload both the .hea and .dat files.")
    sig = _to_standard_order(rec.p_signal, rec.sig_name)
    return sig, int(rec.fs)


def read_upload(files, fs=None):
    """
    files: list of (filename, bytes). Returns (signal of shape (samples, 12), sampling rate).
    """
    if not files:
        raise ECGReadError("No file uploaded.")
    names = [f[0].lower() for f in files]

    if any(n.endswith(".hea") for n in names):
        sig, fs = _read_wfdb(files)
    elif names[0].endswith(".csv") or names[0].endswith(".txt"):
        sig, fs = _read_csv(files[0][1], fs)
    else:
        raise ECGReadError("Unsupported file type. Upload a 12-lead CSV, or a WFDB .hea + .dat pair.")

    sig = np.nan_to_num(sig.astype(float))
    duration = len(sig) / fs
    if not 5 <= duration <= 60:
        raise ECGReadError(f"Recording is {duration:.1f} s long; expected 5-60 s (10 s is standard).")
    if np.allclose(sig.std(axis=0), 0):
        raise ECGReadError("The signal is flat on every lead.")
    return sig, fs


def plot_data(sig, fs, target_fs=250):
    """Downsampled leads for the frontend's ECG viewer."""
    step = max(1, int(fs // target_fs))
    return {
        "fs": fs / step,
        "duration_s": round(len(sig) / fs, 2),
        "units": "mV",
        "leads": [{"name": DISPLAY_NAMES[i], "values": np.round(sig[::step, i], 4).tolist()}
                  for i in range(12)],
    }
