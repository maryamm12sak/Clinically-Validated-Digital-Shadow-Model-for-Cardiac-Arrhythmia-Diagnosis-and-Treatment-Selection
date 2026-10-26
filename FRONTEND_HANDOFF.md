# Frontend Handoff: ECG Arrhythmia Decision Support (FYP)

I'm building the **frontend** for my final-year project. The backend is already built and working. This document describes everything the frontend needs. All endpoint names, field names and values below are real, taken from the backend code.

---

## 1. Project overview

**What the app does:** a doctor uploads a standard 12-lead ECG (10 seconds) and enters a few patient details. The app returns:
- the most likely **heart rhythm** (e.g. atrial fibrillation) with a **confidence score**
- automatic **measurements**: heart rate, PR, QRS, QT, QTc
- an **urgency triage colour**: `green` / `amber` / `red`
- guideline-based **next steps** (what a cardiologist would typically recommend)
- **warnings** (e.g. dangerously long QTc)
- the **12-lead ECG plot** data for display

**Who uses it:** emergency-room doctors and general physicians in Pakistani hospitals that have **no on-site cardiologist**. Prospective clinical partner: Dow University Hospital, Karachi. It is **decision support, not a diagnosis**: the doctor stays responsible for every decision, and a disclaimer must always be shown.

**Main features:**
1. Upload an ECG (CSV, or WFDB `.hea` + `.dat`) and get a full analysis
2. Patient context (age, sex, systolic BP, symptoms, stable or not) that changes the recommendations (e.g. an unstable patient with AF gets cardioversion instead of drugs)
3. 10 built-in demo ECGs (normal, AF, flutter, blocks, etc.) that can be analysed with one click
4. History of all past analyses (audit log)
5. Doctor feedback: agree with or correct each result, plus an overall agreement rate

**Team:** Maryam Sultan (backend), Ali Hamza (ML model), Maryam binte Shahid (frontend). Supervisor: Dr. Imran Khan, IBA Karachi. MVP demo: **30 October 2026**. Final: December 2026.

---

## 2. Tech stack

### Backend (done)
| Part | Choice |
|---|---|
| Language | Python 3.14 |
| Web framework | FastAPI, run with Uvicorn |
| Database | SQLite (single file `ecg.db`, via Python's built-in `sqlite3`) |
| ECG reading | `wfdb` (WFDB files), `numpy` (CSV) |
| ECG measurements | `neurokit2` + custom multi-lead / median-beat code |
| Rules | `rules.yaml`, read with `pyyaml` 6.0.3 |
| File uploads | `python-multipart` |
| Auth | **None** (see section 4) |

Exact versions of every package are in `requirements.txt` in the repo.

### Frontend (to build)
Planned in the project architecture:
- **React + Vite**
- **Plotly** (`react-plotly.js` / `plotly.js`) for the 12-lead ECG viewer with the standard ECG-paper grid
- Plain `fetch` (or axios) for API calls

---

## 3. API reference

**Base URL (local):** `http://127.0.0.1:8000`
**Interactive test page:** `http://127.0.0.1:8000/docs` (the root `/` redirects there)
**Authentication:** no endpoint requires authentication (none is implemented).

### General rules
- All responses are **JSON**, except `GET /samples/{sample_id}/file` (returns a CSV file).
- Requests with data use **form fields** (`multipart/form-data` when a file is included; `application/x-www-form-urlencoded` or `multipart/form-data` for feedback). **Not JSON bodies.**
- All form values are sent as **text**. **Blank fields mean "not given"**, so it's safe to send empty strings.
- **Error format** (all errors):
  ```json
  { "detail": "Human-readable message to show the user" }
  ```
  - `400`: invalid input (bad file, unknown symptom, non-numeric age, etc.). `detail` is a readable string.
  - `404`: unknown sample id or analysis id
  - `422`: a required field is missing, or a path/query parameter has the wrong type (FastAPI's automatic validation). Here `detail` is a **list** of objects, not a string.
  - `500`: unexpected server error (see known limitations)

---

### `GET /health`
Server up? Model loaded?

**Response 200**
```json
{ "status": "ok", "model_loaded": false, "model_version": null }
```

---

### `GET /meta`
Static info the UI needs: labels, allowed symptoms, disclaimer, model info.

**Response 200**
```json
{
  "labels": [
    {"id": "NSR", "name": "Normal sinus rhythm"},
    {"id": "SBRAD", "name": "Sinus bradycardia"},
    {"id": "STACH", "name": "Sinus tachycardia"},
    {"id": "AFIB", "name": "Atrial fibrillation"},
    {"id": "AFLT", "name": "Atrial flutter"},
    {"id": "SVT", "name": "Supraventricular tachycardia"},
    {"id": "AVB1", "name": "First-degree AV block"},
    {"id": "AVB_HIGH", "name": "Second/third-degree AV block"},
    {"id": "LBBB", "name": "Left bundle branch block"},
    {"id": "RBBB", "name": "Right bundle branch block"},
    {"id": "PVC", "name": "Premature ventricular complexes"},
    {"id": "PAC", "name": "Premature atrial complexes"}
  ],
  "model_version": null,
  "metrics": null,
  "rules_version": "draft-0.1",
  "detection_threshold": 0.5,
  "symptoms": ["chest_pain", "dyspnea", "palpitations", "syncope", "dizziness",
               "altered_mental_status", "shock", "heart_failure"],
  "disclaimer": "Decision support only. Draft recommendations pending clinical review. The treating physician is responsible for all decisions."
}
```
`metrics` will hold the model's per-class test results once the model is plugged in (format not fixed yet; handle `null`).

---

### `POST /analyze`
Full analysis of one ECG. **Every call is saved to the database** and returns an `analysis_id`.

**Request:** `multipart/form-data`

| Field | Required | Type / format | Example | Notes |
|---|---|---|---|---|
| `file` | **yes** | file | `ecg.csv` | A 12-lead CSV, **or** the `.hea` file of a WFDB record |
| `file2` | no | file | `rec.dat` | Only for WFDB: the matching `.dat` file |
| `fs` | no | number (Hz) | `500` | Sampling rate. Leave blank for 10-second CSVs (computed as rows ÷ 10) |
| `age` | no | integer | `58` | |
| `sex` | no | text | `male` | `male` / `female` |
| `systolic_bp` | no | integer (mmHg) | `85` | **Below 90 makes the patient "unstable"** |
| `symptoms` | no | comma-separated text | `chest_pain,dyspnea` | Values must come from `/meta` `symptoms` |
| `stable` | no | `true` / `false` | `false` | Clinician's judgement. `false` makes the patient unstable |
| `test_rhythm` | no | label id | `AFIB` | **TESTING ONLY**: forces the placeholder model to "find" this rhythm, to see every UI state. Case-insensitive. Hide in production. |

**Example request (JavaScript)**
```js
const form = new FormData();
form.append("file", fileInput.files[0]);
form.append("age", "58");
form.append("sex", "male");
form.append("systolic_bp", "85");
form.append("symptoms", ["chest_pain"].join(","));
form.append("stable", "");            // blank = not given
const res = await fetch(`${API}/analyze`, { method: "POST", body: form });
const data = await res.json();
if (!res.ok) showError(data.detail);
```
Do **not** set the `Content-Type` header manually; the browser adds the multipart boundary.

**Response 200** (shortened; real field names)
```json
{
  "analysis_id": 12,
  "filenames": ["afib.csv"],
  "context": {"age": 58, "sex": "male", "systolic_bp": 85, "symptoms": ["chest_pain"], "stable": null},
  "model": {
    "loaded": false,
    "version": null,
    "note": "Placeholder rhythm probabilities: the trained model is not plugged in yet."
  },
  "rhythms": [
    {"id": "AFIB", "name": "Atrial fibrillation", "probability": 0.95},
    {"id": "NSR", "name": "Normal sinus rhythm", "probability": 0.02}
  ],
  "measurements": {"heart_rate": 79, "pr_ms": null, "qrs_ms": 60, "qt_ms": 340, "qtc_ms": 389},
  "measurement_notes": [
    "PR not reported: there are no organised P waves in atrial fibrillation/flutter.",
    "QT/QTc are approximate because the rhythm is irregular."
  ],
  "assessment": {
    "triage": "red",
    "primary": "AFIB",
    "stability": {"stable": false, "reasons": ["Systolic BP 85 mmHg (below 90)"]},
    "findings": [
      {
        "id": "AFIB",
        "name": "Atrial fibrillation",
        "probability": 0.95,
        "triage": "red",
        "next_steps": [
          "Synchronized cardioversion (ACLS: 120-200 J biphasic for atrial fibrillation).",
          "Sedate before cardioversion if the patient is conscious and time allows.",
          "Start or plan anticoagulation around cardioversion."
        ]
      }
    ],
    "warnings": [
      {"triage": "amber", "message": "QTc 512 ms (500 or more): high risk of torsades de pointes. Avoid QT-prolonging drugs; check and correct potassium and magnesium."}
    ],
    "rules_version": "draft-0.1",
    "disclaimer": "Decision support only. Draft recommendations pending clinical review. The treating physician is responsible for all decisions."
  },
  "ecg": {
    "fs": 100,
    "duration_s": 10,
    "units": "mV",
    "leads": [
      {"name": "I",   "values": [-0.021, -0.059, -0.074, "... 1000 numbers"]},
      {"name": "II",  "values": ["..."]},
      "... 12 leads in order: I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6"
    ]
  }
}
```

**Field meanings**
- `rhythms`: **all 12 labels**, sorted by probability, highest first. Probabilities are 0–1.
- `measurements`: integers in ms (`heart_rate` in bpm). **Any value can be `null`**: either not measurable, or not applicable to the rhythm (e.g. PR in AF, PR in high-grade block, QT/QTc in flutter). Show `null` as "—" and display `measurement_notes` underneath.
- `assessment.triage`: the **overall** colour (the worst across findings and warnings). Use this as the main badge.
- `assessment.primary`: id of the top finding.
- `assessment.findings`: every rhythm with probability ≥ `detection_threshold` (0.5), highest first, each with its own `triage` and `next_steps`. If nothing reaches 0.5, there is one finding with id **`UNCERTAIN`** (name "Uncertain - needs expert review", triage amber or red).
- `assessment.stability.reasons`: why the patient counts as unstable (empty list if stable).
- `assessment.warnings`: measurement-based alerts, each with its own triage colour (`green` = informational note).
- `ecg`: plot data. `values[k]` is the voltage (mV) at time `k / fs` seconds. `fs` is 250 for 500 Hz uploads (downsampled) and 100 for 100 Hz uploads.
- The response is large (about 200 KB) because of `ecg`.

**Errors**
| Status | When | Example `detail` |
|---|---|---|
| 400 | wrong file type | `"Unsupported file type. Upload a 12-lead CSV, or a WFDB .hea + .dat pair."` |
| 400 | CSV not numeric | `"Could not read the CSV. It should contain only numbers, one column per lead."` |
| 400 | wrong lead count | `"Expected 12 leads, found 8."` |
| 400 | duration out of range | `"Recording is 3.0 s long; expected 5-60 s (10 s is standard)."` |
| 400 | flat signal | `"The signal is flat on every lead."` |
| 400 | WFDB missing `.dat` / `.hea` | `"WFDB upload needs the .hea file together with its .dat file."` |
| 400 | unknown symptom | `"Unknown symptoms ['headache']. Allowed: [...]"` |
| 400 | non-numeric number field | `"age must be a number."` |
| 400 | bad `test_rhythm` | `"test_rhythm must be one of [...]"` |
| 422 | `file` missing | FastAPI validation list |

---

### `POST /parse`
Reads the file and returns **plot data only** (no analysis, nothing saved). Useful for previewing an upload before analysing.

**Request:** `multipart/form-data` with `file` (required), `file2` (optional, WFDB `.dat`), `fs` (optional). Same rules as `/analyze`.

**Response 200**
```json
{ "filenames": ["ecg.csv"], "ecg": { "fs": 100, "duration_s": 10, "units": "mV", "leads": [ ... ] } }
```
**Errors:** same file errors as `/analyze` (400), 422 if `file` is missing.

---

### `GET /samples`
Lists the 10 curated demo ECGs (from the PTB-XL public dataset, test fold).

**Response 200**
```json
{
  "samples": [
    {
      "id": "afib",
      "description": "Atrial fibrillation",
      "file": "afib.csv",
      "source": "PTB-XL ecg_id 7221",
      "ptbxl_codes": ["AFIB"],
      "fs": 100,
      "age": "90+",
      "sex": "male"
    }
  ]
}
```
Sample ids: `normal`, `sinus_brady`, `sinus_tachy`, `afib`, `aflutter`, `avb1`, `avb3`, `lbbb`, `rbbb`, `pvc`.
**Note:** `age` is an **integer or the string `"90+"`** (the dataset hides exact ages over 89).

---

### `GET /samples/{sample_id}`
One demo ECG: its details plus plot data.

**Example:** `GET /samples/afib`
**Response 200:** the sample's fields (as above) plus `"ecg": { "fs", "duration_s", "units", "leads" }`.
**Errors:** `404` with `"No sample called 'xyz'."`

---

### `GET /samples/{sample_id}/file`
Downloads the demo ECG's **CSV file** (`text/csv`). Used to analyse a sample: fetch the file, then send it to `/analyze`.

```js
const blob = await (await fetch(`${API}/samples/afib/file`)).blob();
const form = new FormData();
form.append("file", blob, "afib.csv");   // the filename must end in .csv
await fetch(`${API}/analyze`, { method: "POST", body: form });
```
**Errors:** `404` for an unknown id.

---

### `GET /analyses`
History of past analyses, newest first.

**Query params:** `limit` (integer, default 50, clamped 1–500), `include_tests` (boolean, default `true`; `false` hides runs that used `test_rhythm`).

**Example:** `GET /analyses?limit=20&include_tests=false`

**Response 200**
```json
{
  "analyses": [
    {
      "id": 1,
      "created_at": "2026-10-24T17:16:57+00:00",
      "filename": "normal.csv",
      "primary_rhythm": "NSR",
      "triage": "green",
      "measurements": {"heart_rate": 61, "pr_ms": 180, "qrs_ms": 80, "qt_ms": 370, "qtc_ms": 372},
      "context": {"age": null, "sex": null, "systolic_bp": null, "symptoms": [], "stable": null},
      "is_test": false,
      "feedback": {
        "id": 1, "analysis_id": 1, "created_at": "2026-10-24T17:19:25+00:00",
        "agrees": true, "correct_rhythm": null, "comment": null, "clinician": null
      }
    }
  ]
}
```
- `created_at` is **UTC** (ISO 8601). Convert to local time (Pakistan = UTC+5) for display.
- `feedback` is the **latest** feedback for that analysis, or `null` if none.
- `primary_rhythm` is a label id (map it to a name using `/meta` labels; `UNCERTAIN` is also possible).

**Errors:** `422` if `limit` / `include_tests` have the wrong type.

---

### `GET /analyses/{analysis_id}`
One past analysis, with all its feedback.

**Example:** `GET /analyses/1`
**Response 200**
```json
{
  "id": 1,
  "created_at": "2026-10-24T17:16:57+00:00",
  "filename": "normal.csv",
  "is_test": false,
  "context": { "...": "..." },
  "result": { "...": "the full /analyze response WITHOUT analysis_id, filenames, context and ecg" },
  "feedback": [
    {"id": 1, "analysis_id": 1, "created_at": "...", "agrees": true, "correct_rhythm": null, "comment": null, "clinician": null}
  ]
}
```
`result` contains `model`, `rhythms`, `measurements`, `measurement_notes`, `assessment`. **The ECG plot data is not stored**, so a past analysis cannot be re-plotted.

**Errors:** `404` with `"No analysis with id 999."`; `422` if the id isn't an integer.

---

### `POST /analyses/{analysis_id}/feedback`
A doctor agrees with or corrects a result. Multiple feedback entries per analysis are allowed.

**Request:** form fields (`application/x-www-form-urlencoded` or `multipart/form-data`)

| Field | Required | Format | Notes |
|---|---|---|---|
| `agrees` | **yes** | `true` / `false` (also accepts `yes`/`no`, `1`/`0`) | |
| `correct_rhythm` | no | label id, e.g. `AFLT` | The doctor's rhythm when disagreeing. Must be one of the 12 label ids (case-insensitive). `UNCERTAIN` is not accepted. |
| `comment` | no | text | |
| `clinician` | no | text | Name or initials (free text, not verified) |

**Example**
```js
const form = new FormData();
form.append("agrees", "false");
form.append("correct_rhythm", "AFLT");
form.append("comment", "Flutter waves visible in II");
form.append("clinician", "Dr. A");
await fetch(`${API}/analyses/12/feedback`, { method: "POST", body: form });
```
**Response 200**
```json
{ "feedback_id": 3, "analysis_id": 12 }
```
**Errors:** `400` `"agrees must be true or false."`; `400` `"correct_rhythm must be one of [...]"`; `404` `"No analysis with id 12."`; `422` if `agrees` is missing.

---

### `GET /feedback/summary`
How often doctors agreed with the tool. Counts only **real** (non-test) analyses, using the **latest** feedback per analysis.

**Response 200**
```json
{ "reviewed": 20, "agreed": 18, "agreement_rate": 0.9 }
```
`agreement_rate` is `null` when `reviewed` is 0.

---

## 4. Authentication flow

**There is no authentication.** No login, signup, users, roles, tokens, sessions or cookies exist in the backend. Every endpoint is open. Don't build login screens against the API; there's nothing to call.

The only "identity" is the optional free-text `clinician` field on feedback.

This is acceptable for the MVP (local demo on public data). Before any real hospital use, auth would need to be added on the backend (e.g. login with JWT access tokens in an `Authorization: Bearer` header, plus clinician/admin roles). That is **future work and not designed yet**, so the frontend shouldn't assume a particular scheme.

---

## 5. Data models

### Stored in the database (SQLite, file `ecg.db`)

**`analyses`**: one row per `/analyze` call

| Field | Type | Notes |
|---|---|---|
| `id` | integer, primary key | auto-increment; returned as `analysis_id` |
| `created_at` | text (ISO 8601 UTC) | |
| `filename` | text | uploaded file name(s), comma-joined |
| `context` | JSON text | `{age, sex, systolic_bp, symptoms, stable}` |
| `primary_rhythm` | text | label id or `UNCERTAIN` |
| `triage` | text | `green` / `amber` / `red` |
| `measurements` | JSON text | `{heart_rate, pr_ms, qrs_ms, qt_ms, qtc_ms}` |
| `result` | JSON text | full response minus `ecg` |
| `model_version` | text, nullable | |
| `rules_version` | text | e.g. `draft-0.1` |
| `is_test` | integer 0/1 | 1 if `test_rhythm` was used |

**`feedback`**: one row per doctor review

| Field | Type | Notes |
|---|---|---|
| `id` | integer, primary key | returned as `feedback_id` |
| `analysis_id` | integer, foreign key → `analyses.id` | |
| `created_at` | text (ISO 8601 UTC) | |
| `agrees` | integer 0/1 (returned as boolean) | |
| `correct_rhythm` | text, nullable | label id |
| `comment` | text, nullable | |
| `clinician` | text, nullable | |

**Relationship:** one analysis → zero or many feedback rows.

**Not stored:** the ECG signal, plot data, or any patient name or ID (privacy by design).

### Not stored (static or computed)
- **Label:** `{id, name}`, the 12 rhythms listed in `/meta` (plus `UNCERTAIN` in results)
- **Sample:** `{id, description, file, source, ptbxl_codes, fs, age, sex}` from `samples/index.json`
- **Context:** `{age: int|null, sex: string|null, systolic_bp: int|null, symptoms: string[], stable: bool|null}`
- **Measurements:** `{heart_rate, pr_ms, qrs_ms, qt_ms, qtc_ms}`, each `int|null`
- **Finding:** `{id, name, probability, triage, next_steps: string[]}`
- **Warning:** `{triage, message}`
- **ECG plot:** `{fs, duration_s, units: "mV", leads: [{name, values: number[]}]}`

---

## 6. Setup details

- **Repo:** https://github.com/maryamm12sak/Clinically-Validated-Digital-Shadow-Model-for-Cardiac-Arrhythmia-Diagnosis-and-Treatment-Selection (private; ask Maryam Sultan for access)
- **Backend URL / port:** `http://127.0.0.1:8000`
- **CORS:** open to all origins (`allow_origins=["*"]`, all methods and headers), so the Vite dev server (default `http://localhost:5173`) can call the API directly. No proxy needed.
- **Backend environment variables:** none.
- **Suggested frontend env var:** in `.env` → `VITE_API_BASE_URL=http://127.0.0.1:8000`, read as `import.meta.env.VITE_API_BASE_URL`.

**Run the backend locally (Windows PowerShell):**
```
git clone <repo-url>
cd <repo-folder>
py -m venv fypenv
fypenv\Scripts\activate
pip install -r requirements.txt
python make_samples.py        # only if the samples/ folder is missing; downloads 10 demo ECGs (needs internet; it can be slow)
uvicorn main:app --reload
```
If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
`ecg.db` is created automatically on first run (it's git-ignored, so each person has their own history).

**Backend files (for reference):** `main.py` (endpoints), `ecg_io.py` (file reading, plot data), `ecg_features.py` (measurements), `predictor.py` (model; placeholder), `rules.py` + `rules.yaml` (triage and next steps), `labels.py` (labels, symptoms), `db.py` (SQLite), `API.md` (short API doc).

---

## 7. Validation rules and business logic

### Upload
- Accepted: `.csv` / `.txt` (12 numeric columns; optional header row with lead names I, II, III, aVR, aVL, aVF, V1–V6 in any order), or a WFDB `.hea` + `.dat` pair (`.hea` in `file`, `.dat` in `file2`).
- Recording length must be **5–60 seconds** (10 s is standard). Without `fs`, a CSV is assumed to be exactly 10 s.
- Not supported yet: **XML** (hospital ECG machines) and **images/photos** of ECGs. Accept only `.csv`, `.txt`, `.hea`, `.dat` in the file picker.

### Patient context
- All optional. Send blank for "not given".
- `symptoms`: multi-select limited to the 8 values from `/meta`. Send them comma-separated.
- `sex`: `male` / `female`.
- `stable`: tri-state: true / false / not given.
- **Unstable** if any of: `stable` = false, `systolic_bp` < 90, or symptoms include `altered_mental_status`, `shock` or `heart_failure`. This switches recommendations to the emergency (ACLS) versions and usually makes triage red.
- Symptoms also act as modifiers: e.g. `chest_pain` with LBBB, `chest_pain`/`dyspnea` with RBBB, or `syncope` with PVCs raise the triage to amber and add a step.

### Status values
- Triage: `green` (routine) < `amber` (needs attention) < `red` (urgent). The overall triage is the worst one.
- Rhythm ids: `NSR`, `SBRAD`, `STACH`, `AFIB`, `AFLT`, `SVT`, `AVB1`, `AVB_HIGH`, `LBBB`, `RBBB`, `PVC`, `PAC`, plus `UNCERTAIN` in results only.
- `detection_threshold` = 0.5: a rhythm counts as found at probability ≥ 0.5.

### Must-show items (clinical safety)
- **Always display `assessment.disclaimer`** with every result.
- When `model.loaded` is `false`, show `model.note` as a visible banner (the rhythm is a placeholder).
- Show `measurement_notes` next to the measurements, and render `null` measurements as "—", not 0.
- Show `assessment.stability.reasons` when the patient is unstable.
- Next steps are labelled "draft, pending clinical review"; keep that wording visible.

### Roles and permissions
None (no auth). Anyone can analyse, view history and give feedback.

### Frontend quirk to avoid
Never send the literal text `"string"` in fields (the backend treats it as blank because the `/docs` page fills it in, so it's harmless, but don't rely on it).

---

## 8. Current status

### Finished and tested
- All endpoints above work and were tested.
- CSV and WFDB input; lead reordering; input validation with clear errors.
- Measurements validated against 10 ECGs with known diagnoses (e.g. brady HR 48, first-degree block PR 230, LBBB QRS 130, RBBB QRS 140, normal all in range).
- Rhythm-aware measurements (PR hidden in AF/flutter/high-grade block; QT hidden in flutter).
- Rules engine: triage + next steps for 12 rhythms × stable/unstable, symptom and heart-rate modifiers, QTc / HR / wide-QRS warnings.
- SQLite history + doctor feedback + agreement summary.
- 10 demo samples.

### Incomplete
- **Rhythm prediction is a placeholder.** It always returns NSR 0.91 / AFIB 0.04 / SBRAD 0.03 (or the forced `test_rhythm`). Ali's trained model (ONNX) will replace it. The **response format will not change**, but `model.loaded` will become `true`, and `model_version` and `/meta` `metrics` will be filled in.
- **Rules are a draft**, written from AHA 2020 ACLS and the 2023 ACC/AHA AF guideline, and not yet reviewed by doctors.
- **No authentication.**
- **No XML input** (waiting to learn Dow's ECG machine brand) and **no image/photo upload** (digitization; planned after the MVP).
- **No saliency/heatmap** ("which leads drove the prediction") yet. This will come with the model, and the response may gain a field for it.

### Known limitations and bugs
- An ECG with no detectable heartbeats, or very unusual signals, can cause a **500 error** (the measurement step isn't fully guarded). Show a generic "Could not analyse this ECG" message for 500s.
- Measurements on 100 Hz recordings have about ±10 ms resolution.
- Past analyses can't be re-plotted (signal not stored).
- `/analyze` responses are ~200 KB; avoid keeping many in memory.
- `created_at` is UTC.
- Sample `age` can be the string `"90+"`.
- SQLite is local to whoever runs the server (each teammate has a separate history).

---

## 9. Suggested frontend pages

### 1. Analyze (upload + context): main page
- File picker (`.csv`, `.txt`, `.hea`, `.dat`; second picker or multi-select for WFDB `.dat`).
- Optional preview of the uploaded ECG → `POST /parse`.
- Patient context form: age, sex, systolic BP, symptoms (checkboxes from `/meta` `symptoms`), stable (Yes / No / Not assessed).
- Dev-only `test_rhythm` dropdown (labels from `/meta`), hidden in production.
- Submit → `POST /analyze` → go to Results.
- **Endpoints:** `GET /meta`, `POST /parse`, `POST /analyze`

### 2. Results
- Large **triage badge** (`assessment.triage`), primary rhythm name + confidence.
- Placeholder-model banner if `model.loaded` is false.
- **12-lead ECG viewer** from `ecg` (Plotly; standard layout of 12 leads or 3×4 + rhythm strip; ECG paper grid: 1 small square = 40 ms × 0.1 mV, 1 large square = 200 ms × 0.5 mV; 25 mm/s, 10 mm/mV).
- Measurements panel (HR, PR, QRS, QT, QTc; "—" for null) + `measurement_notes`.
- Findings cards: name, probability, per-finding triage, `next_steps` list.
- Warnings list, with colours.
- Stability box (stable / unstable + `reasons`).
- All rhythm probabilities (bar chart from `rhythms`).
- Disclaimer (always).
- **Feedback widget:** Agree / Disagree (+ correct rhythm dropdown, comment, clinician) → `POST /analyses/{analysis_id}/feedback`.
- **Endpoints:** uses the `/analyze` response; `POST /analyses/{analysis_id}/feedback`; `GET /meta` for label names.

### 3. Demo samples gallery
- Cards for the 10 samples (description, source, age, sex), each with a mini ECG preview.
- "Analyse this sample" → download the CSV, then send it to `/analyze` (optionally with context).
- **Endpoints:** `GET /samples`, `GET /samples/{sample_id}`, `GET /samples/{sample_id}/file`, `POST /analyze`

### 4. History
- Table: date (local time), file, primary rhythm, triage badge, HR, feedback status (agreed / disagreed / not reviewed).
- Toggle "show test runs" (`include_tests`).
- Click a row → Analysis detail.
- **Endpoints:** `GET /analyses`, `GET /meta`

### 5. Analysis detail
- Same layout as Results **without** the ECG plot (not stored), plus a list of all feedback entries and the feedback form.
- **Endpoints:** `GET /analyses/{analysis_id}`, `POST /analyses/{analysis_id}/feedback`

### 6. About / status (or header indicator)
- Server and model status → `GET /health` (e.g. green dot "Server online · Model: placeholder").
- Doctor agreement rate → `GET /feedback/summary`.
- Model version, test metrics, rules version, disclaimer → `GET /meta`.
- Short explanation of the project and its limitations.

### Global
- A shared API helper that reads `VITE_API_BASE_URL`, parses JSON, and shows `detail` on errors (string for 400/404; list for 422; generic message for 500).
- Triage colour tokens: green / amber / red, used consistently.
