# Backend integration

The backend calls `predictor.predict(sig, fs, test_rhythm=None) -> {label_id: probability}`
(`main.py`), with `sig` of shape `(samples, 12)` in mV, leads in `labels.py` order, from `ecg_io.read_upload`.
`ecg_model.predict` has exactly that contract and returns **calibrated** probabilities.

## 1. Where the package lives

Keep it in `ml/deploy/` and import it as a package from the repo root (`ml` and `ml/deploy` are
namespace packages, no `__init__.py` needed):

```python
from ml.deploy import ecg_model
```

This works when the backend is started from the repo root (`uvicorn main:app`); `predictor.py`
also puts its own folder on `sys.path`, so it works from other working directories too. Nothing is
copied, so there is one source of truth for the model files, and the package names (`preprocess`,
`features`) cannot clash with backend modules.

## 2. predictor.py

```python
import logging
import sys
import traceback
from pathlib import Path

from labels import LABEL_IDS

log = logging.getLogger("predictor")
sys.path.insert(0, str(Path(__file__).resolve().parent))


class ModelUnavailableError(RuntimeError):
    """The trained model could not be loaded; no rhythm analysis is possible."""


MODEL_LOADED, MODEL_VERSION, METRICS, MODEL_ERROR = False, None, None, None
THRESHOLDS, EXPLORATORY = {}, set()
try:
    from ml.deploy import ecg_model
    _model = ecg_model.get_model("resnet_only")          # loads ONNX files now, not on first request
    ...                                                  # set MODEL_VERSION, METRICS, THRESHOLDS, EXPLORATORY
    MODEL_LOADED = True
except Exception as e:
    MODEL_ERROR = f"{type(e).__name__}: {e}"
    log.error("ECG model failed to load:\n%s", traceback.format_exc())


def predict(sig, fs, test_rhythm=None):
    if test_rhythm:
        return {lid: (0.95 if lid == test_rhythm else 0.02) for lid in LABEL_IDS}
    if not MODEL_LOADED:
        raise ModelUnavailableError(MODEL_ERROR or "model not loaded")
    return _model.predict(sig, fs)
```

- **If the model files fail to load, the app still starts** (`MODEL_LOADED = False`, error logged
  with traceback), but `/analyze` returns **HTTP 503** with
  `{"detail": "Model unavailable: <reason>. No rhythm analysis was performed."}`. It never shows
  placeholder numbers as if they were results. `/health` and `/meta` report `model_loaded: false`
  and `model_error`. The `test_rhythm` testing path keeps working.
- `METRICS` comes from `manifest.json` -> `test_metrics.resnet_only` (PTB-XL fold 10).

## 3. Thresholds

`rules.yaml` has a single `detection_threshold` (0.5). The model's calibrated probabilities need
**per-label thresholds** (`predictor.THRESHOLDS`: max-F1 thresholds from fold 9, with the
high-sensitivity threshold for AFIB and LBBB). `rules.detected_rhythms` and `rules.assess` take an
optional `thresholds` dict; `main.py` passes `predictor.THRESHOLDS` when the real model ran and
keeps 0.5 for the `test_rhythm` path.

Exploratory labels (`AFLT`, `SVT`, `AVB_HIGH`) above threshold stay in the findings, marked
`"exploratory": true` with a note, and add a warning: "unvalidated, needs expert review".

## 4. requirements.txt

Add `onnxruntime==1.31.1` (numpy, scipy and pyyaml are already listed). Only the optional
`ensemble` mode would also need `lightgbm==4.7.0` (neurokit2 and PyWavelets are already listed).

## 5. Ensemble upgrade (not shipped)

`ecg_model.get_model("ensemble")` averages ResNet1d with LightGBM on 500 Hz hand-crafted features
(fold-10 macro-AUROC 0.9768 vs 0.9709). It costs ~120 ms extra CPU per ECG and needs the
`lightgbm` dependency; with inputs below 500 Hz it returns a warning that the feature path used
upsampled data.
