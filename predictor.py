"""
Rhythm prediction.

PLACEHOLDER until Ali's trained model (ONNX) is ready. When it is:
1. put the .onnx file in this folder,
2. import Ali's preprocess.py here,
3. replace the body of predict() with: preprocess -> onnxruntime session.run -> sigmoid,
4. set MODEL_LOADED = True and MODEL_VERSION.
Nothing else in the backend needs to change.
"""
from labels import LABEL_IDS

MODEL_LOADED = False
MODEL_VERSION = None
METRICS = None   # per-class test metrics from Ali, shown on /meta

_PLACEHOLDER = {"NSR": 0.91, "AFIB": 0.04, "SBRAD": 0.03}


def predict(sig, fs, test_rhythm=None):
    """Returns {label_id: probability} for every label."""
    if test_rhythm:   # testing only: pretend the model found this rhythm
        return {lid: (0.95 if lid == test_rhythm else 0.02) for lid in LABEL_IDS}
    return {lid: _PLACEHOLDER.get(lid, 0.01) for lid in LABEL_IDS}
