import io
import numpy as np
import neurokit2 as nk
from fastapi import FastAPI, UploadFile, File, HTTPException

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

    # REAL: heart rate from lead II (assuming 100 readings/sec for now)
    lead2 = data[:, 1]
    signals, info = nk.ecg_process(lead2, sampling_rate=100)
    heart_rate = round(float(signals["ECG_Rate"].mean()))

    return {
        "filename": file.filename,
        "rhythms": {"Normal sinus rhythm": 0.91, "Atrial fibrillation": 0.04},  # still FAKE until Ali's model
        "heart_rate": heart_rate,                                               # REAL
        "triage": "green",                                                      # FAKE for now
        "next_step": "No urgent action needed. (Draft, pending clinical review)"
    }