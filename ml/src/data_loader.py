"""
PTB-XL loading: metadata, signal cache, per-record validation and fold splits.

Signals are cached once per sampling rate as a float32 .npy of shape (N, T, 12), in the row order
of ptbxl_database.csv (sorted by ecg_id). Nothing is dropped or clipped silently: invalid records
are listed in the validation report with reason codes and only excluded when the config says so.
"""
import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
import yaml

ML_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ML_DIR / "configs" / "benchmark.yaml"

EXPECTED_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
EXTREME_MV = 10.0

# Validation reason codes
BAD_SHAPE = "BAD_SHAPE"
BAD_LEADS = "BAD_LEADS"
HAS_NAN = "HAS_NAN"
HAS_INF = "HAS_INF"
FLAT_LEAD = "FLAT_LEAD"
EXTREME_AMPLITUDE = "EXTREME_AMPLITUDE"
READ_ERROR = "READ_ERROR"


class LockedSplitError(PermissionError):
    """Raised when fold 10 is requested without allow_test=True."""


def load_config(path=None):
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class Split:
    name: str
    meta: pd.DataFrame          # rows of ptbxl_database.csv, index ecg_id
    X: np.ndarray | None        # (N, T, 12) float32, or None if signals were not requested


def validate_record(sig, sig_names, expected_len):
    """Return (reasons, max_abs, flat_leads) for one record. Does not modify the signal."""
    reasons, flat = [], []
    if sig.shape != (expected_len, 12):
        reasons.append(BAD_SHAPE)
    if [n.upper() for n in sig_names] != EXPECTED_LEADS:
        reasons.append(BAD_LEADS)
    if np.isnan(sig).any():
        reasons.append(HAS_NAN)
    if np.isinf(sig).any():
        reasons.append(HAS_INF)
    finite = np.where(np.isfinite(sig), sig, 0.0)
    if finite.size:
        ptp = finite.max(axis=0) - finite.min(axis=0)
        flat = [sig_names[j] for j in np.where(ptp == 0)[0]]
        max_abs = float(np.abs(finite).max())
    else:
        max_abs = float("nan")
    if flat:
        reasons.append(FLAT_LEAD)
    if max_abs > EXTREME_MV:
        reasons.append(EXTREME_AMPLITUDE)
    return reasons, max_abs, flat


class PTBXL:
    def __init__(self, config=None, sampling_rate=None):
        self.config = config or load_config()
        data_cfg = self.config["data"]
        self.root = Path(data_cfg["data_root"])
        self.cache_dir = Path(data_cfg["cache_dir"])
        self.sampling_rate = sampling_rate or data_cfg["sampling_rate"]
        if self.sampling_rate not in (100, 500):
            raise ValueError("PTB-XL is only available at 100 or 500 Hz")

        self.db = pd.read_csv(self.root / "ptbxl_database.csv", index_col="ecg_id").sort_index()
        self.db["scp_codes"] = self.db["scp_codes"].apply(ast.literal_eval)
        self.scp = pd.read_csv(self.root / "scp_statements.csv", index_col=0)
        self._signals = None
        self._report = None

    # ---- signals and validation -------------------------------------------------------------

    @property
    def n_samples(self):
        return 10 * self.sampling_rate

    def _cache_paths(self):
        stem = f"ptbxl_{self.sampling_rate}hz"
        return (self.cache_dir / f"{stem}_float32.npy",
                self.cache_dir / f"{stem}_ecg_ids.npy",
                self.cache_dir / f"{stem}_validation.csv")

    def _cache_is_valid(self):
        sig_path, ids_path, rep_path = self._cache_paths()
        if not (sig_path.exists() and ids_path.exists() and rep_path.exists()):
            return False
        ids = np.load(ids_path)
        if len(ids) != len(self.db) or not np.array_equal(ids, self.db.index.values):
            return False
        arr = np.load(sig_path, mmap_mode="r")
        return arr.shape == (len(self.db), self.n_samples, 12)

    def build_cache(self, force=False, progress=True):
        """Read every record with wfdb into a float32 memmap. Reused if present and complete."""
        if not force and self._cache_is_valid():
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        sig_path, ids_path, rep_path = self._cache_paths()
        col = "filename_lr" if self.sampling_rate == 100 else "filename_hr"
        tmp_path = sig_path.with_suffix(".partial.npy")
        out = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float32,
                                        shape=(len(self.db), self.n_samples, 12))
        rows = []
        for i, (ecg_id, rel) in enumerate(self.db[col].items()):
            try:
                sig, meta = wfdb.rdsamp(str(self.root / rel))
                reasons, max_abs, flat = validate_record(sig, meta["sig_name"], self.n_samples)
                if BAD_SHAPE not in reasons:
                    out[i] = sig.astype(np.float32)
                else:
                    out[i] = np.nan
            except Exception as e:
                reasons, max_abs, flat = [READ_ERROR], float("nan"), []
                out[i] = np.nan
                print(f"read error {ecg_id}: {e!r}")
            rows.append({"ecg_id": ecg_id, "valid": not reasons, "reasons": ";".join(reasons),
                         "max_abs_mv": max_abs, "flat_leads": ";".join(flat)})
            if progress and (i + 1) % 2000 == 0:
                print(f"  cache {self.sampling_rate} Hz: {i + 1}/{len(self.db)}", flush=True)
        out.flush()
        del out
        tmp_path.replace(sig_path)
        np.save(ids_path, self.db.index.values)
        pd.DataFrame(rows).to_csv(rep_path, index=False)

    def signals(self):
        """All signals as a read-only memmap (N, T, 12), rows aligned with self.db."""
        if self._signals is None:
            self.build_cache()
            self._signals = np.load(self._cache_paths()[0], mmap_mode="r")
        return self._signals

    def validation_report(self):
        """DataFrame indexed by ecg_id: valid, reasons (';'-joined codes), max_abs_mv, flat_leads."""
        if self._report is None:
            self.build_cache()
            rep = pd.read_csv(self._cache_paths()[2], index_col="ecg_id")
            rep[["reasons", "flat_leads"]] = rep[["reasons", "flat_leads"]].fillna("")
            self._report = rep
        return self._report

    # ---- splits -----------------------------------------------------------------------------

    def split_folds(self, name):
        splits = self.config["splits"]
        if name not in ("train", "val", "test"):
            raise ValueError(f"unknown split {name!r}")
        return list(splits[name])

    def get_split(self, name, allow_test=False, load_signals=True):
        if name == "test" and self.config["splits"].get("test_locked", True) and not allow_test:
            raise LockedSplitError("The test split (fold 10) is locked. Pass allow_test=True "
                                     "only for the final, pre-registered evaluation.")
        mask = self.db["strat_fold"].isin(self.split_folds(name)).values

        if self.config["data"].get("exclude_invalid", False):
            mask &= self.validation_report().loc[self.db.index, "valid"].values.astype(bool)

        dev = self.config.get("dev_subset", {})
        if dev.get("enabled", False) and name in dev.get("apply_to", ["train", "val", "test"]):
            rng = np.random.default_rng(dev.get("seed", 0))
            idx = np.flatnonzero(mask)
            keep = rng.choice(idx, size=max(1, int(round(len(idx) * dev["fraction"]))), replace=False)
            mask = np.zeros_like(mask)
            mask[np.sort(keep)] = True

        meta = self.db[mask]
        X = np.asarray(self.signals()[np.flatnonzero(mask)]) if load_signals else None
        return Split(name, meta, X)


def assert_no_patient_leakage(db, config):
    """Raise AssertionError if any patient_id appears in more than one of train/val/test."""
    splits = config["splits"]
    groups = {name: set(db.loc[db["strat_fold"].isin(splits[name]), "patient_id"])
              for name in ("train", "val", "test")}
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = groups[a] & groups[b]
            assert not shared, f"{len(shared)} patients shared between {a} and {b}"
