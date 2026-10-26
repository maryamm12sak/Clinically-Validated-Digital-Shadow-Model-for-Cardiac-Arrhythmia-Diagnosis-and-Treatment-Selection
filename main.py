"""
ECG decision-support backend. Run with:  uvicorn main:app --reload
Test page: http://127.0.0.1:8000/docs   |   API description: API.md
"""
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse

import db
import predictor
from ecg_features import compute_features
from ecg_io import ECGReadError, plot_data, read_upload
from labels import LABEL_IDS, LABELS, SYMPTOMS
from rules import RULES, adjust_measurements, assess, detected_rhythms

SAMPLES_DIR = Path(__file__).parent / "samples"

app = FastAPI(title="ECG Arrhythmia Decision Support API", version="0.3.0")
db.init_db()   # creates ecg.db on first run

# Lets the React app (running on a different port) call this server during development
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------- shared logic ----------

async def _read_files(*uploads):
    """Read the uploaded files, ignoring empty fields."""
    out = []
    for f in uploads:
        if f is None or isinstance(f, str) or not getattr(f, "filename", None):
            continue
        out.append((f.filename, await f.read()))
    if not out:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    return out


def _blank(value):
    """Blank, or the word 'string' that the /docs test page fills in as an example."""
    return value is None or str(value).strip().lower() in ("", "string")


def _text(value):
    return None if _blank(value) else str(value).strip()


def _num(value, name, cast=float):
    """Form values arrive as text; blank means 'not given'."""
    if _blank(value):
        return None
    try:
        return cast(value)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"{name} must be a number.")


def _bool(value):
    """true/false from a form field; None if blank or not recognised."""
    if _blank(value):
        return None
    v = str(value).strip().lower()
    if v in ("true", "1", "yes", "y"):
        return True
    if v in ("false", "0", "no", "n"):
        return False
    return None


def _load(files, fs=None):
    try:
        return read_upload(files, fs)
    except ECGReadError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _run_analysis(sig, fs, context, test_rhythm=None):
    probabilities = predictor.predict(sig, fs, test_rhythm=test_rhythm)
    measurements, measurement_notes = adjust_measurements(compute_features(sig, fs),
                                                          detected_rhythms(probabilities))
    rhythms = sorted(
        [{"id": lid, "name": l["name"], "probability": round(probabilities[lid], 3)}
         for lid, l in zip(LABEL_IDS, LABELS)],
        key=lambda r: r["probability"], reverse=True)
    return {
        "model": {"loaded": predictor.MODEL_LOADED, "version": predictor.MODEL_VERSION,
                  "note": None if predictor.MODEL_LOADED else
                  "Placeholder rhythm probabilities: the trained model is not plugged in yet."},
        "rhythms": rhythms,
        "measurements": measurements,
        "measurement_notes": measurement_notes,
        "assessment": assess(probabilities, measurements, context),
        "ecg": plot_data(sig, fs),
    }


def _sample_index():
    index_file = SAMPLES_DIR / "index.json"
    if not index_file.exists():
        return []
    return json.loads(index_file.read_text(encoding="utf-8"))


def _get_sample(sample_id):
    for s in _sample_index():
        if s["id"] == sample_id:
            return s
    raise HTTPException(status_code=404, detail=f"No sample called '{sample_id}'.")


# ---------- endpoints ----------

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    """Is the server up, and is the model loaded?"""
    return {"status": "ok", "model_loaded": predictor.MODEL_LOADED, "model_version": predictor.MODEL_VERSION}


@app.get("/meta")
def meta():
    """Label list, model version, test metrics, and the options the frontend form needs."""
    return {
        "labels": [{"id": l["id"], "name": l["name"]} for l in LABELS],
        "model_version": predictor.MODEL_VERSION,
        "metrics": predictor.METRICS,
        "rules_version": RULES["version"],
        "detection_threshold": RULES["detection_threshold"],
        "symptoms": SYMPTOMS,
        "disclaimer": RULES["disclaimer"],
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(..., description="ECG file: a 12-lead CSV, or the .hea file of a WFDB record"),
    file2: UploadFile = File(None, description="Only for WFDB: the matching .dat file"),
    fs: Optional[str] = Form(None, description="Sampling rate (Hz). Leave empty for 10-second CSVs."),
    age: Optional[str] = Form(None),
    sex: Optional[str] = Form(None, description="male / female"),
    systolic_bp: Optional[str] = Form(None, description="mmHg"),
    symptoms: Optional[str] = Form(None, description="Comma-separated, e.g. chest_pain,dyspnea"),
    stable: Optional[str] = Form(None, description="Clinician's judgement: true / false"),
    test_rhythm: Optional[str] = Form(None, description="TESTING ONLY: pretend the model found this rhythm id, e.g. AFIB"),
):
    """Full analysis: rhythm probabilities, measurements, triage, next steps and plot data."""
    test_rhythm = (_text(test_rhythm) or "").upper() or None
    symptoms, sex = _text(symptoms), _text(sex)
    if test_rhythm and test_rhythm not in LABEL_IDS:
        raise HTTPException(status_code=400, detail=f"test_rhythm must be one of {LABEL_IDS}")
    symptom_list = [s.strip() for s in symptoms.split(",") if s.strip()] if symptoms else []
    unknown = [s for s in symptom_list if s not in SYMPTOMS]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown symptoms {unknown}. Allowed: {SYMPTOMS}")

    raw = await _read_files(file, file2)
    sig, sample_rate = _load(raw, _num(fs, "fs"))
    context = {"age": _num(age, "age", int), "sex": sex.lower() if sex else None,
               "systolic_bp": _num(systolic_bp, "systolic_bp", int),
               "symptoms": symptom_list, "stable": _bool(stable)}
    result = _run_analysis(sig, sample_rate, context, test_rhythm)
    filenames = [r[0] for r in raw]
    analysis_id = db.save_analysis(", ".join(filenames), context, result, is_test=bool(test_rhythm))
    return {"analysis_id": analysis_id, "filenames": filenames, "context": context, **result}


@app.post("/parse")
async def parse(
    file: UploadFile = File(..., description="ECG file: a 12-lead CSV, or the .hea file of a WFDB record"),
    file2: UploadFile = File(None, description="Only for WFDB: the matching .dat file"),
    fs: Optional[str] = Form(None),
):
    """Reads the file and returns plot data only, with no analysis (for previewing an upload)."""
    raw = await _read_files(file, file2)
    sig, sample_rate = _load(raw, _num(fs, "fs"))
    return {"filenames": [r[0] for r in raw], "ecg": plot_data(sig, sample_rate)}


@app.get("/samples")
def list_samples():
    """The curated demo ECGs."""
    return {"samples": _sample_index()}


@app.get("/samples/{sample_id}")
def get_sample(sample_id: str):
    """One demo ECG: its description and plot data."""
    s = _get_sample(sample_id)
    sig, fs = _load([(s["file"], (SAMPLES_DIR / s["file"]).read_bytes())])
    return {**s, "ecg": plot_data(sig, fs)}


@app.get("/samples/{sample_id}/file")
def get_sample_file(sample_id: str):
    """Download the demo ECG's CSV (so the frontend can send it to /analyze)."""
    s = _get_sample(sample_id)
    return FileResponse(SAMPLES_DIR / s["file"], media_type="text/csv", filename=s["file"])


# ---------- history and feedback (SQLite) ----------

@app.get("/analyses")
def list_analyses(limit: int = 50, include_tests: bool = True):
    """Past analyses, newest first (the audit log). Set include_tests=false to hide test_rhythm runs."""
    return {"analyses": db.list_analyses(min(max(limit, 1), 500), include_tests)}


@app.get("/analyses/{analysis_id}")
def get_analysis(analysis_id: int):
    """One past analysis: context, the full result (without plot data) and any doctor feedback."""
    a = db.get_analysis(analysis_id)
    if a is None:
        raise HTTPException(status_code=404, detail=f"No analysis with id {analysis_id}.")
    return a


@app.post("/analyses/{analysis_id}/feedback")
def add_feedback(
    analysis_id: int,
    agrees: str = Form(..., description="Does the doctor agree with the tool? true / false"),
    correct_rhythm: Optional[str] = Form(None, description="If not, the doctor's rhythm id, e.g. AFLT"),
    comment: Optional[str] = Form(None),
    clinician: Optional[str] = Form(None, description="Name or initials of the reviewing doctor"),
):
    """Doctor confirms or corrects the tool's result."""
    agree = _bool(agrees)
    if agree is None:
        raise HTTPException(status_code=400, detail="agrees must be true or false.")
    correct = (_text(correct_rhythm) or "").upper() or None
    if correct and correct not in LABEL_IDS:
        raise HTTPException(status_code=400, detail=f"correct_rhythm must be one of {LABEL_IDS}")
    fid = db.add_feedback(analysis_id, agree, correct, _text(comment), _text(clinician))
    if fid is None:
        raise HTTPException(status_code=404, detail=f"No analysis with id {analysis_id}.")
    return {"feedback_id": fid, "analysis_id": analysis_id}


@app.get("/feedback/summary")
def feedback_summary():
    """How often doctors agreed with the tool (real analyses only, latest feedback per analysis)."""
    return db.feedback_summary()
