import { getFeedbackSummary } from "../api.js";
import { Card, ErrorBox, Spinner } from "../components/ui.jsx";
import { useFetch } from "../hooks.js";
import { useMeta } from "../state.jsx";

export default function About() {
  const { meta, metaError, health, disclaimer, reloadMeta } = useMeta();
  const summary = useFetch(getFeedbackSummary, []);
  const s = summary.data;

  return (
    <div className="stack">
      <div>
        <h1>About this tool</h1>
        <p className="muted">
          Decision support for emergency-room and general doctors in hospitals without an on-site cardiologist. Upload a
          12-lead ECG and get the likely rhythm, measurements, an urgency colour and guideline-based next steps.
        </p>
      </div>

      <div className="alert alert-amber">{disclaimer}</div>

      <div className="two-col">
        <Card title="System status">
          <dl className="facts">
            <dt>Server</dt>
            <dd>
              {health.state === "online" && <span className="chip chip-green">Online</span>}
              {health.state === "offline" && <span className="chip chip-red">Offline</span>}
              {health.state === "checking" && "Checking…"}
            </dd>
            <dt>Rhythm model</dt>
            <dd>
              {health.state !== "online"
                ? "—"
                : health.model_loaded
                  ? `Loaded${health.model_version ? ` (${health.model_version})` : ""}`
                  : "Placeholder (trained model not plugged in yet)"}
            </dd>
            <dt>Rules version</dt>
            <dd>{meta?.rules_version ?? "—"}</dd>
            <dt>Detection threshold</dt>
            <dd>{meta ? `${Math.round(meta.detection_threshold * 100)}%` : "—"}</dd>
          </dl>
          {metaError && <ErrorBox message={metaError} onRetry={reloadMeta} />}
        </Card>

        <Card title="Doctor agreement">
          {summary.loading && <Spinner label="Loading…" />}
          {summary.error && <ErrorBox message={summary.error.message} onRetry={summary.reload} />}
          {s &&
            (s.reviewed === 0 ? (
              <p className="muted">No real (non-test) analyses have been reviewed yet.</p>
            ) : (
              <>
                <p className="big-number">{Math.round(s.agreement_rate * 100)}%</p>
                <p className="muted">
                  Doctors agreed with {s.agreed} of {s.reviewed} reviewed analyses (test runs excluded).
                </p>
              </>
            ))}
        </Card>
      </div>

      <Card title="Model test results">
        {meta?.metrics ? (
          <pre className="pre">{JSON.stringify(meta.metrics, null, 2)}</pre>
        ) : (
          <p className="muted">Not available yet. They will appear here once the trained model is plugged in.</p>
        )}
      </Card>

      <Card title="Limitations">
        <ul className="notes">
          <li>The rhythm result is currently a placeholder until the trained model is connected.</li>
          <li>Next steps are drafts written from AHA/ACLS guidance and have not yet been reviewed by doctors.</li>
          <li>Only 12-lead CSV/TXT and WFDB (.hea + .dat) files are supported. No XML or photos yet.</li>
          <li>On 100 Hz recordings, intervals are accurate to about ±10 ms.</li>
          <li>The ECG signal is not stored, so past analyses can't be re-plotted.</li>
          <li>There is no login: anyone with access to the app can analyse, view history and give feedback.</li>
        </ul>
      </Card>
    </div>
  );
}
