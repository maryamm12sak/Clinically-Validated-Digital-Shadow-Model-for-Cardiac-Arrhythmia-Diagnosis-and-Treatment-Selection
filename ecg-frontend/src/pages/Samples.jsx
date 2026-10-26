import { useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { analyze, getSample, getSampleFile, listSamples } from "../api.js";
import { Card, ErrorBox, Sparkline, Spinner } from "../components/ui.jsx";
import { useFetch } from "../hooks.js";
import { useLatest, useMeta } from "../state.jsx";
import { EMPTY_CONTEXT, SAMPLE_LABEL, SHOW_TEST_TOOLS } from "../util.js";

function SampleCard({ sample, forceExpected, busyId, onAnalyse, onWithContext }) {
  const ref = useRef(null);
  const [lead, setLead] = useState(null);
  const [previewError, setPreviewError] = useState(false);

  // Load the preview signal only when the card scrolls into view.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ctrl = new AbortController();
    const io = new IntersectionObserver(([entry]) => {
      if (!entry.isIntersecting) return;
      io.disconnect();
      getSample(sample.id, ctrl.signal)
        .then((d) => setLead(d.ecg.leads.find((l) => l.name === "II")?.values ?? d.ecg.leads[0]?.values ?? null))
        .catch((e) => e.name !== "AbortError" && setPreviewError(true));
    });
    io.observe(el);
    return () => {
      io.disconnect();
      ctrl.abort();
    };
  }, [sample.id]);

  const busy = busyId === sample.id;
  const age = sample.age ?? "—";
  return (
    <article className="card sample-card" ref={ref}>
      <div className="spark-wrap">
        {previewError ? <div className="spark spark-empty">Preview unavailable</div> : <Sparkline values={lead} />}
        <small className="muted">Lead II</small>
      </div>
      <h3>{sample.description}</h3>
      <p className="muted small">
        {sample.sex ? `${sample.sex[0].toUpperCase()}${sample.sex.slice(1)}` : "—"}, age {age} · {sample.source}
      </p>
      <div className="row">
        <button type="button" className="btn btn-small" disabled={Boolean(busyId)} onClick={() => onAnalyse(sample)}>
          {busy ? "Analysing…" : "Analyse this sample"}
        </button>
        <button type="button" className="btn btn-small btn-ghost" disabled={Boolean(busyId)} onClick={() => onWithContext(sample)}>
          With patient context…
        </button>
      </div>
      {forceExpected && <small className="muted">Will force “{SAMPLE_LABEL[sample.id] ?? "—"}” (test run)</small>}
    </article>
  );
}

export default function Samples() {
  const navigate = useNavigate();
  const { setLatest } = useLatest();
  const { health } = useMeta();
  const { loading, data, error, reload } = useFetch(listSamples, []);
  const [busyId, setBusyId] = useState(null);
  const [runError, setRunError] = useState(null);
  const [force, setForce] = useState(false);

  const placeholderModel = health.state === "online" && health.model_loaded === false;
  const showForce = SHOW_TEST_TOOLS && placeholderModel;
  const testRhythmFor = (s) => (showForce && force ? (SAMPLE_LABEL[s.id] ?? "") : "");
  const sexOf = (s) => (s.sex === "male" || s.sex === "female" ? s.sex : "");

  const analyseNow = async (s) => {
    setBusyId(s.id);
    setRunError(null);
    try {
      const blob = await getSampleFile(s.id);
      const file = new File([blob], s.file?.endsWith(".csv") ? s.file : `${s.id}.csv`, { type: "text/csv" });
      const testRhythm = testRhythmFor(s);
      const context = { ...EMPTY_CONTEXT, age: Number.isInteger(s.age) ? String(s.age) : "", sex: sexOf(s) };
      const result = await analyze({ files: { main: file, dat: null }, fs: "", context, testRhythm });
      setLatest({ id: result.analysis_id, result, testRhythm });
      navigate(`/results/${result.analysis_id}`);
    } catch (e) {
      setRunError(e.message);
      setBusyId(null);
    }
  };

  const withContext = (s) =>
    navigate("/", { state: { sampleId: s.id, age: s.age, sex: sexOf(s), testRhythm: testRhythmFor(s) } });

  return (
    <div className="stack">
      <div>
        <h1>Demo ECGs</h1>
        <p className="muted">Ten public PTB-XL recordings with known diagnoses. Analyse one with a single click.</p>
      </div>

      {showForce && (
        <label className="check test-toggle">
          <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} />
          <span>
            <strong>Testing only:</strong> force each sample's expected rhythm (the model is still a placeholder, so
            otherwise every sample returns “Normal sinus rhythm”).
          </span>
        </label>
      )}

      {runError && <ErrorBox title="Analysis failed" message={runError} />}
      {loading && <Spinner label="Loading samples…" />}
      {error && <ErrorBox message={error.message} onRetry={reload} />}

      {data && (
        <div className="sample-grid">
          {data.samples.map((s) => (
            <SampleCard
              key={s.id}
              sample={s}
              forceExpected={showForce && force}
              busyId={busyId}
              onAnalyse={analyseNow}
              onWithContext={withContext}
            />
          ))}
        </div>
      )}
      {!loading && !error && data?.samples.length === 0 && (
        <Card>
          <p className="muted">No samples found. Run <code>python make_samples.py</code> in the backend folder.</p>
        </Card>
      )}
    </div>
  );
}
