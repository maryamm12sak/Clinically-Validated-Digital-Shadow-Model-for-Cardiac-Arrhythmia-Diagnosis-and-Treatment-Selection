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
    </div>
  );
}
