import { useMeta } from "../state.jsx";
import { pct, symptomLabel } from "../util.js";
import EcgViewer from "./EcgViewer.jsx";
import { Card, TriageBadge } from "./ui.jsx";

const MEASURES = [
  ["heart_rate", "Heart rate", "bpm"],
  ["pr_ms", "PR", "ms"],
  ["qrs_ms", "QRS", "ms"],
  ["qt_ms", "QT", "ms"],
  ["qtc_ms", "QTc", "ms"],
];

function ContextChips({ context }) {
  if (!context) return null;
  const chips = [];
  if (context.age != null) chips.push(`Age ${context.age}`);
  if (context.sex) chips.push(context.sex[0].toUpperCase() + context.sex.slice(1));
  if (context.systolic_bp != null) chips.push(`SBP ${context.systolic_bp} mmHg`);
  if (context.stable === true) chips.push("Marked stable");
  if (context.stable === false) chips.push("Marked unstable");
  (context.symptoms ?? []).forEach((s) => chips.push(symptomLabel(s)));
  if (!chips.length) return <small className="muted">No patient context given</small>;
  return (
    <div className="chips">
      {chips.map((c) => (
        <span key={c} className="chip">
          {c}
        </span>
      ))}
    </div>
  );
}

/**
 * Shared by the Results page (with ECG signal) and Analysis detail (without).
 * `result` = the /analyze response shape (the stored version lacks ecg/context/etc.).
 */
export default function ResultView({ result, context, ecg, filename, testRhythm, isTest }) {
  const { labelName, threshold, disclaimer: metaDisclaimer } = useMeta();
  const { model, rhythms = [], measurements = {}, measurement_notes: notes = [], assessment } = result;
  const findings = assessment?.findings ?? [];
  const warnings = assessment?.warnings ?? [];
  const stability = assessment?.stability;
  const primary = findings.find((f) => f.id === assessment?.primary) ?? findings[0];
  const uncertain = assessment?.primary === "UNCERTAIN";
  const top = rhythms[0];
  const disclaimer = assessment?.disclaimer || metaDisclaimer;
  const forced = testRhythm || isTest;

  return (
    <div className="stack">
      {forced && (
        <div className="alert alert-purple" role="status">
          <strong>Test run.</strong> The rhythm was forced{testRhythm ? ` to “${labelName(testRhythm)}”` : ""} to exercise
          the interface. This is not a model prediction.
        </div>
      )}
      {model && model.loaded === false && (
        <div className="alert alert-amber" role="status">
          <strong>Placeholder rhythm result.</strong> {model.note}
        </div>
      )}

      <Card className="summary">
        <div className="summary-grid">
          <div className="summary-triage">
            <TriageBadge level={assessment?.triage} size="lg" />
            <small className="muted">Overall urgency</small>
          </div>
          <div className="summary-main">
            <small className="muted">{uncertain ? "Result" : "Most likely rhythm"}</small>
            <h1>{primary?.name ?? labelName(assessment?.primary)}</h1>
            {uncertain ? (
              top && (
                <p className="muted">
                  Highest model probability: {top.name} ({pct(top.probability)}), below the {pct(threshold)} detection
                  threshold.
                </p>
              )
            ) : (
              <p className="confidence">
                <strong>{pct(primary?.probability)}</strong> confidence
              </p>
            )}
          </div>
          <div className="summary-meta">
            {filename && <div className="muted">{filename}</div>}
            <ContextChips context={context} />
          </div>
        </div>
        <p className="disclaimer">{disclaimer}</p>
      </Card>

      {stability && stability.stable === false && (
        <div className="alert alert-red" role="status">
          <strong>Patient treated as unstable</strong>
          {stability.reasons?.length > 0 && (
            <ul>
              {stability.reasons.map((r) => (
                <li key={r}>{r}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {ecg ? (
        <Card title="12-lead ECG">
          <EcgViewer ecg={ecg} />
        </Card>
      ) : (
        <p className="muted note">The ECG signal isn't stored, so past analyses can't be re-plotted.</p>
      )}

      <div className="two-col">
        <Card title="Measurements">
          <div className="measures">
            {MEASURES.map(([key, label, unit]) => (
              <div key={key} className="measure">
                <small>{label}</small>
                <strong>{measurements[key] ?? "—"}</strong>
                <small>{unit}</small>
              </div>
            ))}
          </div>
          {notes.length > 0 && (
            <ul className="notes">
              {notes.map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
          )}
        </Card>

        <Card title="Warnings">
          {warnings.length === 0 ? (
            <p className="muted">No measurement warnings.</p>
          ) : (
            <ul className="warning-list">
              {warnings.map((w) => (
                <li key={w.message} className={`warning warning-${w.triage}`}>
                  <TriageBadge level={w.triage} size="sm" />
                  <span>{w.message}</span>
                </li>
              ))}
            </ul>
          )}
        </Card>
      </div>

      <Card title="Findings and next steps" aside={<span className="chip chip-purple">Draft · pending clinical review</span>}>
        <div className="stack-sm">
          {findings.map((f) => (
            <article key={f.id} className={`finding finding-${f.triage}`}>
              <header className="row-between">
                <h3>{f.name}</h3>
                <div className="row">
                  {typeof f.probability === "number" && <span className="muted">{pct(f.probability)}</span>}
                  <TriageBadge level={f.triage} size="sm" />
                </div>
              </header>
              {f.next_steps?.length > 0 ? (
                <ol>
                  {f.next_steps.map((s) => (
                    <li key={s}>{s}</li>
                  ))}
                </ol>
              ) : (
                <p className="muted">No next steps listed.</p>
              )}
            </article>
          ))}
        </div>
      </Card>

      <Card title="All rhythm probabilities">
        <ul className="bars">
          {rhythms.map((r) => (
            <li key={r.id}>
              <span className="bar-label">{r.name}</span>
              <span className="bar-track" aria-hidden="true">
                <span className="bar-fill" style={{ width: `${Math.max(0, Math.min(1, r.probability)) * 100}%` }} />
                <span className="bar-threshold" style={{ left: `${threshold * 100}%` }} />
              </span>
              <span className="bar-value">{pct(r.probability)}</span>
            </li>
          ))}
        </ul>
        <small className="muted">The vertical line marks the {pct(threshold)} detection threshold.</small>
      </Card>
    </div>
  );
}
