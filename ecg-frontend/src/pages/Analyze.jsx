import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { analyze, classifyFiles, getSampleFile, parseEcg } from "../api.js";
import ContextForm from "../components/ContextForm.jsx";
import EcgViewer from "../components/EcgViewer.jsx";
import FileDrop from "../components/FileDrop.jsx";
import { Card, ErrorBox, Spinner } from "../components/ui.jsx";
import { useLatest, useMeta } from "../state.jsx";
import { EMPTY_CONTEXT, SHOW_TEST_TOOLS } from "../util.js";

function validate(ctx, fs) {
  const errors = {};
  if (ctx.age !== "" && !(Number.isInteger(Number(ctx.age)) && Number(ctx.age) >= 0 && Number(ctx.age) <= 130))
    errors.age = "Enter a whole number between 0 and 130.";
  if (
    ctx.systolicBp !== "" &&
    !(Number.isInteger(Number(ctx.systolicBp)) && Number(ctx.systolicBp) >= 20 && Number(ctx.systolicBp) <= 300)
  )
    errors.systolicBp = "Enter a whole number between 20 and 300.";
  if (fs !== "" && !(Number(fs) > 0)) errors.fs = "Enter a positive number, or leave blank.";
  return errors;
}

export default function Analyze() {
  const navigate = useNavigate();
  const location = useLocation();
  const { setLatest } = useLatest();
  const { labels } = useMeta();

  const [files, setFiles] = useState([]);
  const [fs, setFs] = useState("");
  const [ctx, setCtx] = useState(EMPTY_CONTEXT);
  const [testRhythm, setTestRhythm] = useState("");
  const [preview, setPreview] = useState({ state: "idle" }); // idle | loading | ok | error
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState(null);
  const [touched, setTouched] = useState(false);

  const cls = useMemo(() => classifyFiles(files), [files]);
  const errors = validate(ctx, fs);
  const fileReady = Boolean(cls.main) && !cls.error;

  // Arrived from the samples gallery ("Analyse with patient context").
  const sampleId = location.state?.sampleId;
  useEffect(() => {
    if (!sampleId) return;
    let alive = true;
    getSampleFile(sampleId)
      .then((blob) => {
        if (!alive) return;
        setFiles([new File([blob], `${sampleId}.csv`, { type: "text/csv" })]);
        const { age, sex } = location.state;
        setCtx((c) => ({ ...c, age: Number.isInteger(age) ? String(age) : "", sex: sex ?? "" }));
        if (SHOW_TEST_TOOLS && location.state.testRhythm) setTestRhythm(location.state.testRhythm);
      })
      .catch((e) => alive && setSubmitError(e.message));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sampleId]);

  // Preview the signal as soon as a valid file combination is chosen (nothing is saved).
  useEffect(() => {
    if (!fileReady || errors.fs) {
      setPreview({ state: "idle" });
      return;
    }
    const ctrl = new AbortController();
    setPreview({ state: "loading" });
    const t = setTimeout(() => {
      parseEcg({ main: cls.main, dat: cls.dat }, fs, ctrl.signal)
        .then((d) => setPreview({ state: "ok", ecg: d.ecg }))
        .catch((e) => e.name !== "AbortError" && setPreview({ state: "error", message: e.message }));
    }, 300);
    return () => {
      clearTimeout(t);
      ctrl.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cls.main, cls.dat, fs, fileReady]);

  const submit = async (e) => {
    e.preventDefault();
    setTouched(true);
    setSubmitError(null);
    if (!fileReady || Object.keys(errors).length) return;
    setSubmitting(true);
    try {
      const result = await analyze({ files: { main: cls.main, dat: cls.dat }, fs, context: ctx, testRhythm });
      setLatest({ id: result.analysis_id, result, testRhythm });
      navigate(`/results/${result.analysis_id}`);
    } catch (err) {
      setSubmitError(err.message);
      setSubmitting(false);
    }
  };

  const fileError = cls.error && (files.length > 0 ? cls.error : null);

  return (
    <form onSubmit={submit} className="stack" noValidate>
      <div>
        <h1>Analyse an ECG</h1>
        <p className="muted">
          Upload a standard 12-lead ECG (10 seconds) and add what you know about the patient. Everything below the file is
          optional.
        </p>
      </div>

      <Card title="1 · ECG file">
        <FileDrop
          files={files}
          onFiles={(f) => {
            setFiles(f);
            setSubmitError(null);
          }}
          error={fileError ?? (touched && !files.length ? "Choose an ECG file first." : null)}
          hint="Accepted: .csv, .txt, or a .hea + .dat pair. XML and photos of ECGs aren't supported yet."
        />

        <details className="advanced">
          <summary>Advanced: sampling rate</summary>
          <label className="field">
            <span>Sampling rate (Hz)</span>
            <input
              type="number"
              min="1"
              value={fs}
              onChange={(e) => setFs(e.target.value)}
              placeholder="Leave blank for 10-second CSVs"
            />
            {errors.fs && <em className="field-error">{errors.fs}</em>}
          </label>
        </details>

        {preview.state === "loading" && <Spinner label="Reading file…" />}
        {preview.state === "error" && <ErrorBox title="This file can't be used" message={preview.message} />}
        {preview.state === "ok" && (
          <div className="preview">
            <small className="muted">Preview. Nothing has been analysed or saved yet.</small>
            <EcgViewer ecg={preview.ecg} />
          </div>
        )}
      </Card>

      <Card title="2 · Patient context (optional)">
        <ContextForm value={ctx} onChange={setCtx} errors={errors} />
      </Card>

      {SHOW_TEST_TOOLS && (
        <Card title="Testing only" className="card-test">
          <label className="field">
            <span>Force a rhythm (placeholder model)</span>
            <select value={testRhythm} onChange={(e) => setTestRhythm(e.target.value)}>
              <option value="">Off: use the model output</option>
              {labels.map((l) => (
                <option key={l.id} value={l.id}>
                  {l.name}
                </option>
              ))}
            </select>
          </label>
          <p className="hint">Hidden when VITE_SHOW_TEST_TOOLS is not "true". Test runs are labelled in the results and history.</p>
        </Card>
      )}

      {submitError && <ErrorBox title="Analysis failed" message={submitError} />}

      <div className="actions">
        <button type="submit" className="btn btn-large" disabled={submitting || (files.length > 0 && !fileReady)}>
          {submitting ? "Analysing…" : "Analyse ECG"}
        </button>
        {submitting && <Spinner label="Measuring and analysing. This can take a few seconds." />}
      </div>
    </form>
  );
}
