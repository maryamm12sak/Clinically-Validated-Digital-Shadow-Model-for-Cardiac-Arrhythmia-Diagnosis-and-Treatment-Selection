import { useMeta } from "../state.jsx";
import { fmtLocal } from "../util.js";

export default function FeedbackList({ entries }) {
  const { labelName } = useMeta();
  if (!entries?.length) return <p className="muted">No feedback has been given for this analysis yet.</p>;
  return (
    <ul className="feedback-list">
      {entries.map((f) => (
        <li key={f.id}>
          <div className="row-between">
            <span className={`chip ${f.agrees ? "chip-green" : "chip-red"}`}>{f.agrees ? "Agreed" : "Disagreed"}</span>
            <small className="muted">
              {fmtLocal(f.created_at)}
              {f.clinician ? ` · ${f.clinician}` : ""}
            </small>
          </div>
          {f.correct_rhythm && <div>Correct rhythm: {labelName(f.correct_rhythm)}</div>}
          {f.comment && <div className="muted">“{f.comment}”</div>}
        </li>
      ))}
    </ul>
  );
}
