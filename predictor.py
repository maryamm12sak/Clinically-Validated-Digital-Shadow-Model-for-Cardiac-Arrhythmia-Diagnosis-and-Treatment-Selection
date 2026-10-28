"""
Rhythm prediction with Ali's trained model (ml/deploy, see ml/deploy/MODEL_CARD.md).

Model: ResNet1d, 3 training seeds averaged, exported to ONNX; returns calibrated probabilities.
It is loaded once at import. If loading fails, the app still starts, but MODEL_LOADED is False,
MODEL_ERROR says why, and predict() raises ModelUnavailableError (main.py turns that into HTTP 503)
instead of returning made-up numbers. The test_rhythm path works either way.
"""
import logging
import sys
import traceback
from pathlib import Path

from labels import LABEL_IDS

log = logging.getLogger("predictor")
sys.path.insert(0, str(Path(__file__).resolve().parent))   # so "ml.deploy" imports from any working dir

MODEL_MODE = "resnet_only"
# per-label threshold policy: max-F1 by default, high-sensitivity (>= 0.90 on fold 9) for these
HIGH_SENSITIVITY_LABELS = {"AFIB", "LBBB"}


class ModelUnavailableError(RuntimeError):
    """The trained model could not be loaded, so no rhythm analysis is possible."""


MODEL_LOADED = False
MODEL_VERSION = None
MODEL_ERROR = None
METRICS = None        # PTB-XL fold-10 (test) metrics, shown on /meta
THRESHOLDS = {}       # {label_id: threshold on the calibrated probability}
EXPLORATORY = set()   # labels with too few training examples to validate: need expert review
_model = None

try:
    from ml.deploy import ecg_model

    _model = ecg_model.get_model(MODEL_MODE)
    _th = _model.calibration["thresholds"]
    THRESHOLDS = {lid: (_th["sens90"][j] if lid in HIGH_SENSITIVITY_LABELS else _th["max_f1"][j])
                  for j, lid in enumerate(_model.label_ids)}
    EXPLORATORY = set(_model.exploratory)
    MODEL_VERSION = f"{_model.manifest['model_version']} ({MODEL_MODE})"
    METRICS = _model.manifest["test_metrics"][MODEL_MODE]
    if _model.label_ids != LABEL_IDS:
        raise RuntimeError(f"model labels {_model.label_ids} do not match labels.py {LABEL_IDS}")
    MODEL_LOADED = True
except Exception as e:
    MODEL_ERROR = f"{type(e).__name__}: {e}"
    log.error("ECG model failed to load; /analyze will return 503.\n%s", traceback.format_exc())
    print(f"*** ECG MODEL NOT LOADED: {MODEL_ERROR} ***", file=sys.stderr, flush=True)


def predict(sig, fs, test_rhythm=None):
    """Returns {label_id: calibrated probability} for every label."""
    if test_rhythm:   # testing only: pretend the model found this rhythm
        return {lid: (0.95 if lid == test_rhythm else 0.02) for lid in LABEL_IDS}
    if not MODEL_LOADED:
        raise ModelUnavailableError(MODEL_ERROR or "the model is not loaded")
    return _model.predict(sig, fs)
