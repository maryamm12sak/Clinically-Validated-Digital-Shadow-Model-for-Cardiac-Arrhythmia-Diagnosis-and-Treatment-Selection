import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException

from ecg_features import compute_features

app = FastAPI()


@app.get("/")
def home():
    return {"message": "ECG server is running"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    content = await file.read()
    try:
        data = np.loadtxt(io.BytesIO(content), delimiter=",", skiprows=1)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read file. Upload a 12-lead ECG CSV.")

    fs = round(len(data) / 10)   # PTB-XL ECGs are 10 seconds long
    features = compute_features(data, fs=fs)   # REAL: heart rate + intervals (whole 12-lead ECG)

    return {
        "filename": file.filename,
        "rhythms": {"Normal sinus rhythm": 0.91, "Atrial fibrillation": 0.04},  # still FAKE until Ali's model
        **features,
        "triage": "green",                                                      # FAKE for now
        "next_step": "No urgent action needed. (Draft, pending clinical review)"
    }
