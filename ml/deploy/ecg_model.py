"""
Deployed PTB-XL arrhythmia model (decision support, not a diagnosis).

    from ecg_model import predict, predict_detailed
    probs = predict(sig, fs)              # {label_id: calibrated probability}, labels.yaml order

sig: numpy array (samples, 12), float, millivolts, leads in the order
     I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6. fs: any sampling rate (Hz). 5-60 s long.

Modes:
  resnet_only  (default, shipped for the MVP) ResNet1d, 3 seeds, ONNX
  ensemble     documented upgrade: mean of ResNet1d and LightGBM (3 seeds, 500 Hz features)
Each mode uses its own calibrators and thresholds, fitted on PTB-XL fold 9.
Dependencies: resnet_only needs numpy, scipy, onnxruntime, pyyaml; ensemble additionally needs
lightgbm, neurokit2 and PyWavelets, which are imported only when ensemble mode is used.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml

try:
    from . import preprocess as pp
except ImportError:                                  # imported as a top-level module
    import preprocess as pp

DEPLOY_DIR = Path(__file__).resolve().parent
MODEL_FS = 100           # ResNet1d input rate
FEATURE_FS = 500         # rate the LightGBM features were computed at (PTB-XL records500)
WINDOW = 250             # 2.5 s at 100 Hz
OVERLAP = 0.5
MIN_SECONDS, MAX_SECONDS = 5.0, 60.0
EPS = 1e-6
MODES = ("resnet_only", "ensemble")
DEFAULT_MODE = "resnet_only"


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _calibrate(p, cal):
    if cal["method"] == "temperature":
        return float(_sigmoid(_logit(p) / cal["T"]))
    if cal["method"] == "platt":
        return float(_sigmoid(cal["a"] * _logit(p) + cal["b"]))
    return float(p)


class ECGModel:
    def __init__(self, mode=DEFAULT_MODE, model_dir=DEPLOY_DIR):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.dir = Path(model_dir)
        self.manifest = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        labels = yaml.safe_load((self.dir / "labels.yaml").read_text(encoding="utf-8"))["labels"]
        self.label_ids = [lab["id"] for lab in labels]
        self.exploratory = {lab["id"] for lab in labels if lab["tier"] == "exploratory"}
        cal_file = "calibration_ensemble.json" if mode == "ensemble" else "calibration_resnet_only.json"
        self.calibration = json.loads((self.dir / cal_file).read_text(encoding="utf-8"))
        assert self.calibration["labels"] == self.label_ids

        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("ECG_MODEL_THREADS", "4"))
        self.sessions, self.normalizers = [], []
        for comp in self.manifest["components"]["resnet1d"]:
            self.sessions.append(ort.InferenceSession(str(self.dir / comp["onnx"]), opts,
                                                      providers=["CPUExecutionProvider"]))
            self.normalizers.append(pp.PerLeadNormalizer.load(self.dir / comp["normalizer"]))

        self.boosters = []
        if mode == "ensemble":
            import lightgbm as lgb
            self.feature_columns = json.loads((self.dir / "feature_columns.json").read_text(encoding="utf-8"))
            for comp in self.manifest["components"]["lightgbm"]:
                seed_models = []
                for lid in self.label_ids:
                    spec = comp["labels"][lid]
                    if "constant" in spec:
                        seed_models.append(float(spec["constant"]))
                    else:
                        seed_models.append(lgb.Booster(model_file=str(self.dir / spec["file"])))
                self.boosters.append(seed_models)

    # ---- components -------------------------------------------------------------------------

    def resnet_probability(self, sig, fs):
        """Seed-averaged sigmoid of the per-label max logit over 2.5 s windows. Shape (12,)."""
        x = pp.preprocess(sig, fs, MODEL_FS)
        probs = []
        for sess, norm in zip(self.sessions, self.normalizers):
            windows, _ = pp.sliding_windows(norm.transform(x), WINDOW, OVERLAP)
            logits = sess.run(None, {"input": np.ascontiguousarray(windows.transpose(0, 2, 1))})[0]
            probs.append(_sigmoid(logits.max(axis=0).astype(np.float64)))
        return np.mean(probs, axis=0)

    def feature_vector(self, sig, fs):
        try:
            from . import features as fx
        except ImportError:
            import features as fx
        x = pp.resample(np.asarray(sig, dtype=np.float64), fs, FEATURE_FS)
        feats = fx.extract_features(x, FEATURE_FS)
        return np.array([[feats[c] for c in self.feature_columns]], dtype=np.float64), feats

    def lightgbm_probability(self, sig, fs):
        X, feats = self.feature_vector(sig, fs)
        per_seed = []
        for seed_models in self.boosters:
            per_seed.append([m if isinstance(m, float) else float(m.predict(X.astype(np.float32))[0])
                             for m in seed_models])
        return np.mean(per_seed, axis=0), feats

    # ---- public API ---------------------------------------------------------------------------

    def _check(self, sig, fs):
        sig = np.asarray(sig, dtype=np.float64)
        if sig.ndim != 2 or sig.shape[1] != 12:
            raise ValueError(f"expected an array of shape (samples, 12), got {sig.shape}")
        if not fs or fs <= 0:
            raise ValueError("sampling rate fs must be positive")
        seconds = len(sig) / fs
        if not MIN_SECONDS <= seconds <= MAX_SECONDS:
            raise ValueError(f"ECG is {seconds:.1f} s long; the model accepts {MIN_SECONDS:.0f}-"
                             f"{MAX_SECONDS:.0f} s")
        if not np.isfinite(sig).all():
            raise ValueError("ECG contains NaN or infinite values")
        return sig, seconds

    def predict_detailed(self, sig, fs):
        t0 = time.perf_counter()
        sig, seconds = self._check(sig, fs)
        warnings_ = []
        p_resnet = self.resnet_probability(sig, fs)
        components = ["resnet1d_x3"]
        p_lgbm, feats = None, None
        if self.mode == "ensemble":
            if fs < FEATURE_FS:
                warnings_.append(f"input sampled at {fs} Hz: the LightGBM feature path used data "
                                 f"upsampled to {FEATURE_FS} Hz (its training features were computed "
                                 f"at {FEATURE_FS} Hz), so the ensemble may be less accurate")
            p_lgbm, feats = self.lightgbm_probability(sig, fs)
            components.append("lightgbm_x3")
            if feats["err_any"]:
                warnings_.append("some ECG features could not be computed (e.g. R peaks not found); "
                                 "LightGBM used missing values for them")
            p_raw = (p_resnet + p_lgbm) / 2
        else:
            p_raw = p_resnet
        th = self.calibration["thresholds"]
        labels = {}
        for j, lid in enumerate(self.label_ids):
            p_cal = _calibrate(p_raw[j], self.calibration["calibrators"][j])
            labels[lid] = {
                "resnet_probability": float(p_resnet[j]),
                "lightgbm_probability": None if p_lgbm is None else float(p_lgbm[j]),
                "raw_probability": float(p_raw[j]),
                "calibrated_probability": p_cal,
                "threshold_max_f1": th["max_f1"][j],
                "threshold_high_sensitivity": th["sens90"][j],
                "detected": p_cal >= th["max_f1"][j],
                "detected_high_sensitivity": p_cal >= th["sens90"][j],
                "exploratory": lid in self.exploratory,
                "note": "needs expert review: too few training examples to validate"
                        if lid in self.exploratory else "",
            }
        return {"model_version": self.manifest["model_version"], "mode": self.mode,
                "components": components, "duration_s": seconds, "fs": fs, "warnings": warnings_,
                "labels": labels, "runtime_ms": (time.perf_counter() - t0) * 1000}

    def predict(self, sig, fs):
        d = self.predict_detailed(sig, fs)
        return {lid: v["calibrated_probability"] for lid, v in d["labels"].items()}


_MODELS = {}


def get_model(mode=DEFAULT_MODE):
    if mode not in _MODELS:
        _MODELS[mode] = ECGModel(mode)
    return _MODELS[mode]


def predict(sig, fs, mode=DEFAULT_MODE):
    """{label_id: calibrated probability} for every label, same signature as predictor.predict."""
    return get_model(mode).predict(sig, fs)


def predict_detailed(sig, fs, mode=DEFAULT_MODE):
    return get_model(mode).predict_detailed(sig, fs)


def metrics(mode=DEFAULT_MODE):
    """Fold-10 (PTB-XL test fold) metrics for the backend's /meta endpoint."""
    return get_model(mode).manifest["test_metrics"][mode]
