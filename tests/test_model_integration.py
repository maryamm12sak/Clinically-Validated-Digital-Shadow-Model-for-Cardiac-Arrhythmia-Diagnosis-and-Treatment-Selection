"""
Backend + trained model, end to end over HTTP.

FastAPI's TestClient needs httpx, which is not in requirements.txt, so these tests start the app with
uvicorn on 127.0.0.1 (temporary SQLite file, so ecg.db is not touched) and call it with requests.
Run from the repo root:  python -m pytest tests -q
"""
import io
import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests
from scipy.signal import resample_poly

REPO = Path(__file__).resolve().parents[1]
SAMPLES = REPO / "samples"

LAUNCHER = """
import sys, uvicorn
sys.path.insert(0, {repo!r})
import db
db.DB_PATH = __import__("pathlib").Path({db!r})
if {broken_dir!r}:
    from ml.deploy import ecg_model
    ecg_model.ECGModel.__init__.__defaults__ = (ecg_model.DEFAULT_MODE, __import__("pathlib").Path({broken_dir!r}))
import main
uvicorn.run(main.app, host="127.0.0.1", port={port}, log_level="warning")
"""


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start(tmp_path, broken_dir=""):
    port = _free_port()
    code = LAUNCHER.format(repo=str(REPO), db=str(tmp_path / "test.db"), broken_dir=str(broken_dir), port=port)
    proc = subprocess.Popen([sys.executable, "-c", code], cwd=str(tmp_path),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    base = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            if requests.get(base + "/health", timeout=1).ok:
                return proc, base
        except requests.ConnectionError:
            time.sleep(0.5)
        if proc.poll() is not None:
            break
    proc.kill()
    raise RuntimeError("server did not start:\n" + (proc.stdout.read() if proc.stdout else ""))


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    proc, base = _start(tmp_path_factory.mktemp("srv"))
    yield base
    proc.kill()


@pytest.fixture(scope="module")
def broken_server(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("broken")
    model_dir = tmp / "deploy"
    shutil.copytree(REPO / "ml" / "deploy", model_dir, ignore=shutil.ignore_patterns("__pycache__"))
    (model_dir / "models" / "resnet1d_seed0.onnx").unlink()        # simulate a missing model file
    proc, base = _start(tmp, model_dir)
    yield base, proc
    proc.kill()


def _analyze(base, name, content, fs=None, **form):
    data = {"fs": str(fs)} if fs else {}
    data.update(form)
    t0 = time.perf_counter()
    r = requests.post(base + "/analyze", files={"file": (name, content, "text/csv")}, data=data, timeout=60)
    return r, (time.perf_counter() - t0) * 1000


def test_health_and_meta(server):
    h = requests.get(server + "/health").json()
    assert h["model_loaded"] is True and h["model_error"] is None
    assert h["model_version"].startswith("ptbxl-arrhythmia-v1.0")
    m = requests.get(server + "/meta").json()
    assert m["metrics"]["macro_auroc_primary"] == pytest.approx(0.9709, abs=1e-4)
    assert "fold 10" in m["metrics"]["dataset"]
    assert set(m["detection_thresholds"]) == {l["id"] for l in m["labels"]}
    assert m["exploratory_labels"] == ["AFLT", "AVB_HIGH", "SVT"]


def test_demo_samples(server):
    index = json.loads((SAMPLES / "index.json").read_text(encoding="utf-8"))
    files = [(s["file"], s["description"]) for s in index] + [("../ecg_00001_500hz.csv", "PTB-XL ecg 1, 500 Hz")]
    report = []
    for fname, expected in files:
        r, ms = _analyze(server, Path(fname).name, (SAMPLES / fname).read_bytes())
        assert r.status_code == 200, (fname, r.text[:300])
        body = r.json()
        assert body["model"]["loaded"] and body["model"]["probabilities"] == "calibrated"
        assert len(body["rhythms"]) == 12
        for f in body["assessment"]["findings"]:
            if f.get("exploratory"):
                assert "expert review" in f["note"]
        report.append({"file": fname, "expected": expected, "latency_ms": round(ms, 1),
                       "top3": [(x["id"], x["probability"], x["probability"] >= x["threshold"])
                                for x in body["rhythms"][:3]],
                       "findings": [f["id"] for f in body["assessment"]["findings"]],
                       "triage": body["assessment"]["triage"]})
    out = REPO / "ml" / "results" / "deploy"
    out.mkdir(parents=True, exist_ok=True)
    (out / "backend_demo.json").write_text(json.dumps(report, indent=1), encoding="utf-8")


def test_250hz_csv(server):
    df = pd.read_csv(SAMPLES / "normal.csv")
    sig = resample_poly(df.values, 5, 2, axis=0)            # 100 Hz -> 250 Hz
    buf = io.StringIO()
    pd.DataFrame(sig, columns=df.columns).to_csv(buf, index=False, float_format="%.5f")
    r, _ = _analyze(server, "normal_250hz.csv", buf.getvalue().encode(), fs=250)
    assert r.status_code == 200, r.text[:300]
    assert r.json()["rhythms"][0]["id"] == "NSR"


def test_too_short_is_400(server):
    df = pd.read_csv(SAMPLES / "normal.csv").iloc[:300]      # 3 s at 100 Hz
    r, _ = _analyze(server, "short.csv", df.to_csv(index=False).encode(), fs=100)
    assert r.status_code == 400 and "5-60" in r.json()["detail"]


def test_test_rhythm_path_unchanged(server):
    r, _ = _analyze(server, "normal.csv", (SAMPLES / "normal.csv").read_bytes(), test_rhythm="AFIB")
    body = r.json()
    assert r.status_code == 200 and body["assessment"]["primary"] == "AFIB"
    assert all(x["threshold"] == 0.5 for x in body["rhythms"])


def test_missing_model_file_gives_503(broken_server):
    base, _ = broken_server
    h = requests.get(base + "/health").json()
    assert h["model_loaded"] is False and "resnet1d_seed0.onnx" in h["model_error"]
    m = requests.get(base + "/meta").json()
    assert m["model_loaded"] is False and m["metrics"] is None
    r, _ = _analyze(base, "afib.csv", (SAMPLES / "afib.csv").read_bytes())
    assert r.status_code == 503
    body = r.json()
    assert body["detail"].startswith("Model unavailable:") and "No rhythm analysis was performed" in body["detail"]
    assert "rhythms" not in body and "assessment" not in body
    r, _ = _analyze(base, "afib.csv", (SAMPLES / "afib.csv").read_bytes(), test_rhythm="AFIB")
    assert r.status_code == 200                                  # testing path still works
