import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { listAnalyses } from "../api.js";
import { Card, ErrorBox, Spinner, TriageBadge } from "../components/ui.jsx";
import { useFetch } from "../hooks.js";
import { useMeta } from "../state.jsx";
import { fmtLocal } from "../util.js";

function FeedbackStatus({ feedback, labelName }) {
  if (!feedback) return <span className="muted">Not reviewed</span>;
  if (feedback.agrees) return <span className="chip chip-green">Agreed</span>;
  return (
    <span className="chip chip-red">
      Disagreed{feedback.correct_rhythm ? ` → ${labelName(feedback.correct_rhythm)}` : ""}
    </span>
  );
}

export default function History() {
  const navigate = useNavigate();
  const { labelName } = useMeta();
  const [showTests, setShowTests] = useState(true);
  const { loading, data, error, reload } = useFetch(() => listAnalyses({ limit: 100, includeTests: showTests }), [showTests]);
  const rows = data?.analyses ?? [];

  return (
    <div className="stack">
      <div className="row-between">
        <div>
          <h1>History</h1>
          <p className="muted">Every analysis is logged here, newest first. Times are Pakistan time.</p>
        </div>
        <label className="check">
          <input type="checkbox" checked={showTests} onChange={(e) => setShowTests(e.target.checked)} />
          Show test runs
        </label>
      </div>

      {loading && <Spinner label="Loading history…" />}
      {error && <ErrorBox message={error.message} onRetry={reload} />}

      {data && rows.length === 0 && (
        <Card>
          <p className="muted">
            No analyses yet. <Link to="/">Analyse an ECG</Link> or try a <Link to="/samples">demo sample</Link>.
          </p>
        </Card>
      )}

      {rows.length > 0 && (
        <Card className="table-card">
          <div className="table-scroll">
            <table className="table">
              <thead>
                <tr>
                  <th>When</th>
                  <th>File</th>
                  <th>Rhythm</th>
                  <th>Urgency</th>
                  <th>HR</th>
                  <th>Feedback</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((a) => (
                  <tr key={a.id} className="clickable" onClick={() => navigate(`/history/${a.id}`)}>
                    <td>
                      <Link to={`/history/${a.id}`} onClick={(e) => e.stopPropagation()}>
                        {fmtLocal(a.created_at)}
                      </Link>
                    </td>
                    <td>
                      {a.filename}
                      {a.is_test && <span className="chip chip-purple">Test</span>}
                    </td>
                    <td>{labelName(a.primary_rhythm)}</td>
                    <td>
                      <TriageBadge level={a.triage} size="sm" />
                    </td>
                    <td>{a.measurements?.heart_rate ?? "—"}</td>
                    <td>
                      <FeedbackStatus feedback={a.feedback} labelName={labelName} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  );
}
