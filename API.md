# ECG Backend API (v0.3)

Base URL (local): `http://127.0.0.1:8000`. Interactive test page: `/docs`. CORS is open for development.

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Server up? Model loaded? |
| GET | `/meta` | Labels, model version, test metrics, allowed symptoms, disclaimer |
| POST | `/analyze` | Full analysis of an uploaded ECG + patient context |
| POST | `/parse` | Plot data only (preview an upload) |
| GET | `/samples` | List demo ECGs |
| GET | `/samples/{id}` | One demo ECG with plot data |
| GET | `/samples/{id}/file` | Download a demo ECG's CSV (send it on to `/analyze`) |
| GET | `/analyses` | History of past analyses, newest first |
| GET | `/analyses/{id}` | One past analysis with its feedback |
| POST | `/analyses/{id}/feedback` | Doctor agrees with or corrects the result |
| GET | `/feedback/summary` | How often doctors agreed with the tool |

## POST /analyze

`multipart/form-data`:

| Field | Required | Example | Notes |
|---|---|---|---|
| `file` | yes | `ecg.csv` | A 12-lead CSV, or the `.hea` file of a WFDB record |
| `file2` | no | `rec.dat` | Only for WFDB: the matching `.dat` file |
| `fs` | no | `500` | Sampling rate; leave empty for 10-second CSVs |
| `age` | no | `58` | |
| `sex` | no | `male` | `male` / `female` |
| `systolic_bp` | no | `85` | mmHg; below 90 makes the patient unstable |
| `symptoms` | no | `chest_pain,dyspnea` | Comma-separated; allowed values from `/meta` |
| `stable` | no | `false` | Clinician's judgement |
| `test_rhythm` | no | `AFIB` | **Testing only**: forces a rhythm so you can see every UI state |

Response (shortened):

```json
{
  "analysis_id": 12,
  "filenames": ["ecg.csv"],
  "context": {"age": 58, "sex": "male", "systolic_bp": 85, "symptoms": ["chest_pain"], "stable": null},
  "model": {"loaded": false, "version": null, "note": "Placeholder rhythm probabilities: ..."},
  "rhythms": [{"id": "NSR", "name": "Normal sinus rhythm", "probability": 0.91}, "... all 12 labels, highest first"],
  "measurements": {"heart_rate": 64, "pr_ms": 139, "qrs_ms": 86, "qt_ms": 379, "qtc_ms": 391},
  "measurement_notes": ["PR not reported: there are no organised P waves in atrial fibrillation/flutter."],
  "assessment": {
    "triage": "red",
    "primary": "AFIB",
    "stability": {"stable": false, "reasons": ["Systolic BP 85 mmHg (below 90)"]},
    "findings": [
      {"id": "AFIB", "name": "Atrial fibrillation", "probability": 0.95, "triage": "red",
       "next_steps": ["Synchronized cardioversion (ACLS: 120-200 J biphasic for atrial fibrillation).", "..."]}
    ],
    "warnings": [{"triage": "amber", "message": "QTc 512 ms (500 or more): ..."}],
    "rules_version": "draft-0.1",
    "disclaimer": "Decision support only. Draft recommendations pending clinical review. ..."
  },
  "ecg": {
    "fs": 250, "duration_s": 10.0, "units": "mV",
    "leads": [{"name": "I", "values": [0.01, 0.02, "..."]}, "... 12 leads: I, II, III, aVR, aVL, aVF, V1-V6"]
  }
}
```

- `triage` is `green`, `amber` or `red`. Show the overall `assessment.triage` as the main badge.
- `findings` lists every rhythm at or above the detection threshold (0.5). If none, one finding with id `UNCERTAIN`.
- Any measurement can be `null` if it couldn't be measured or doesn't apply to the rhythm (e.g. PR in atrial fibrillation). `measurement_notes` explains why; show these under the measurements.
- `ecg.leads[i].values` are in mV at `ecg.fs` samples per second (time of sample k = k / fs). Standard ECG paper: 25 mm/s, 10 mm/mV.

All form fields are sent as text; blank fields are treated as "not given". `/parse` takes the same `file`, `file2` and `fs` fields.

Errors: status `400` with `{"detail": "message to show the user"}` (bad file, unknown symptom), `404` for an unknown sample.

## History and feedback

Every `/analyze` call is saved in a local SQLite database (`ecg.db`) and returns an `analysis_id`.
No ECG signal or patient identifier is stored: only the file name, the clinical context, and the tool's output.
Runs that used `test_rhythm` are marked `is_test: true`.

**GET /analyses?limit=50&include_tests=true**: list of `{id, created_at, filename, primary_rhythm, triage, measurements, context, is_test, feedback}` (feedback = latest, or null).

**GET /analyses/{id}**: `{id, created_at, filename, is_test, context, result, feedback: [...]}`. `result` is the `/analyze` response without `ecg`.

**POST /analyses/{id}/feedback** (form fields):

| Field | Required | Example |
|---|---|---|
| `agrees` | yes | `true` / `false` |
| `correct_rhythm` | no | `AFLT` (label id, when the doctor disagrees) |
| `comment` | no | `flutter waves visible in II` |
| `clinician` | no | `Dr. A` |

Returns `{feedback_id, analysis_id}`.

**GET /feedback/summary**: `{reviewed, agreed, agreement_rate}` over real (non-test) analyses.
