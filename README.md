# Clinically validated digital shadow model for cardiac arrhythmia diagnosis and treatment selection

Final-year project, IBA Karachi. A web app where a clinician uploads a standard 12-lead ECG and gets the
likely heart rhythm (12 labels), automatic measurements (heart rate, PR, QRS, QT, QTc), an urgency colour
and guideline-based next steps.

**This is a decision-support prototype, not a medical device and not a diagnosis.** It has only been
evaluated on public research data (PTB-XL) and is not validated for clinical use.

Team: Maryam Sultan (backend), Ali Hamza (ML model), Maryam binte Shahid (frontend).

## Run it (Windows)

Prerequisites:
- Windows 10/11 (the only platform tested)
- [Python](https://www.python.org/downloads/) 3.13 (tested; other 3.11+ versions may work)
- [Node.js](https://nodejs.org) 18 or newer (tested with 24)
- Internet access on the first run

Then **double-click `run_app.bat`** in this folder.

- **First run:** it creates a Python environment in `.venv` and installs the packages from `requirements.txt`, then runs `npm ci` for the website. This takes several minutes.
- **Later runs:** they skip the install and start in a few seconds.
- **Two windows open:**
  - **ECG backend** (http://127.0.0.1:8000; API test page at http://127.0.0.1:8000/docs)
  - **ECG website** (http://localhost:5173)
- **When the backend is ready**, the launcher prints whether the model loaded and opens the website in your browser.
- **Try it:** the **Samples** page has 10 demo ECGs from the PTB-XL test fold (normal, sinus brady/tachy, AF, flutter, 1st- and 3rd-degree AV block, LBBB, RBBB, PVC), analysed with one click.
- **Stop it** by closing both windows.

The launcher starts the website with `VITE_SHOW_TEST_TOOLS=false`, so the testing-only "force rhythm"
controls are hidden.

Manual start (any OS), from this folder:
```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt      # macOS/Linux: .venv/bin/python
.venv\Scripts\python -m uvicorn main:app --port 8000
cd ecg-frontend && npm ci && npm run dev
```

## The model

- **Where it lives:** the trained model is packaged in `ml/deploy/`. The backend loads it at startup through `predictor.py`; no training, dataset or GPU is needed to run the app.
- **What it is:** ResNet1d, a 1D convolutional network. Three training seeds are averaged, exported to ONNX and run with onnxruntime. It outputs calibrated probabilities for 12 labels, each with its own decision threshold.
- **If the model files are missing:** the app still starts, but analyses return HTTP 503 "Model unavailable" instead of fake results.
- **More detail:** `ml/deploy/MODEL_CARD.md` covers the input/output contract and limitations; `ml/deploy/INTEGRATION.md` covers how it is wired into the backend.

Results on the held-out PTB-XL test fold (fold 10, 2,198 ECGs, evaluated once after pre-registration),
macro-AUROC over the 9 primary labels with 95% patient-level bootstrap CI:

| Model | Macro-AUROC |
|---|---|
| ResNet1d, 3 seeds (**shipped**) | 0.971 [0.966-0.976] |
| ResNet1d + LightGBM ensemble (available as `ensemble` mode, not enabled) | 0.977 [0.973-0.981] |

Known weaknesses:
- **NSR:** low specificity (~0.58).
- **SBRAD:** low sensitivity (~0.44).
- **PAC and AVB1:** about 50% precision.
- **Exploratory labels:** atrial flutter, SVT and 2nd/3rd-degree AV block have too few examples to validate, so the app marks them "needs expert review".

The numbers come from `ml/results/final/` (`test_metrics.csv`, `test_per_class.csv`;
pre-registration in `preregistration.json`). Validation-fold benchmarks of all models are in
`ml/results/benchmark_results.csv` and `ml/results/ensembles/`, and plots are in `ml/results/plots/`.

## Tests

With the app environment plus pytest (`.venv\Scripts\python -m pip install pytest`):
```
.venv\Scripts\python -m pytest tests -q        # backend + model end to end (starts the app on a free port)
```

The ML tests (`ml/tests`) need the ML environment (`ml/requirements-ml.txt`). Most of them also need the
PTB-XL dataset; they are skipped if it is not found.

## Retraining / reproducing the benchmark (optional)

This is not needed to run the app. You need:
- PTB-XL v1.0.3 (~3 GB), by default placed in a `data` folder next to this repo (`../data/ptb-xl-1.0.3`; paths are set in `ml/configs/benchmark.yaml`);
- the packages in `ml/requirements-ml.txt` (CPU PyTorch).

See `ml/README.md`. Entry points: `ml/benchmark.py` (classical and deep models), `ml/analyze_round2.py`
(ensembles, calibration), `ml/export_deploy.py` (builds `ml/deploy`). Trained checkpoints are not in git.

## Data credit

PTB-XL, a large publicly available electrocardiography dataset, v1.0.3. Wagner P, Strodthoff N,
Bousseljot R, Samek W, Schaeffter T. PhysioNet (2022), https://doi.org/10.13026/kfzx-aw45, licensed under
CC BY 4.0. Original publication: Wagner P et al., *Scientific Data* 7, 154 (2020).

## Known limits

- **One dataset only:** trained and tested on PTB-XL (Germany, 1989-1996), with no external or local-hospital validation.
- **Windows only:** run and tested on Windows only.
- **No ventricular tachycardia label.**
- **No login or authentication:** this is a local demo only.
