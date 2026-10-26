# ECG Decision Support: Frontend

React + Vite + Plotly frontend for the FYP backend (FastAPI). No login (the backend has none).

## Run it (Windows PowerShell)

```
npm install
npm run dev          # http://localhost:5173
```

The backend must be running at the URL in `.env` (`VITE_API_BASE_URL`, default `http://127.0.0.1:8000`):

```
uvicorn main:app --reload      # in the backend folder
```

No backend handy? Run the built-in stand-in API instead (same contract, fake clinical content):

```
npm run mock         # http://127.0.0.1:8000, in a second terminal
```

In the mock, a file name containing `bad` returns a 400 and one containing `crash` returns a 500.

## Settings (`.env`)

| Variable | Meaning |
|---|---|
| `VITE_API_BASE_URL` | Backend address |
| `VITE_SHOW_TEST_TOOLS` | `true` shows the testing-only "force rhythm" controls. **Set to `false` for the real demo.** |

## Pages

| Route | What |
|---|---|
| `/` | Upload (CSV or .hea + .dat), live ECG preview via `/parse`, patient context, Analyse |
| `/results/:id` | Triage badge, rhythm + confidence, 12-lead ECG, measurements, warnings, next steps, probabilities, feedback |
| `/samples` | 10 demo ECGs, one-click analysis |
| `/history`, `/history/:id` | Audit log; past result (no ECG plot, it isn't stored) + feedback |
| `/about` | Server/model status, doctor agreement rate, model metrics, limitations |

## Notes

- Always shown: the disclaimer, the placeholder-model banner (while `model.loaded` is false), measurement notes, instability reasons, and the "Draft · pending clinical review" label on next steps.
- When Ali's model is plugged in, nothing needs to change: the banner disappears by itself and the About page fills in version and metrics.
- Errors: 400/404 show the backend's message, 422 is flattened to readable text, 500 on analyse shows "Could not analyse this ECG…".
- Plotly is used directly (`plotly.js-basic-dist-min`) rather than through `react-plotly.js`; same result, one fewer dependency.
