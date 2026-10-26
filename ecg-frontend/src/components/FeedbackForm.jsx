import { useState } from "react";
import { postFeedback } from "../api.js";
import { useMeta } from "../state.jsx";

const CLINICIAN_KEY = "ecg.clinician";
const readClinician = () => {
  try {
    return localStorage.getItem(CLINICIAN_KEY) ?? "";
  } catch {
    return "";
  }
};

export default function FeedbackForm({ analysisId, onSaved }) {
  const { labels } = useMeta();
  const [agrees, setAgrees] = useState(null);
  const [rhythm, setRhythm] = useState("");
  const [comment, setComment] = useState("");
  const [clinician, setClinician] = useState(readClinician);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [saved, setSaved] = useState(false);

  const submit = async (e) => {
    e.preventDefault();
    if (agrees === null) return setError("Choose Agree or Disagree first.");
    setBusy(true);
    setError(null);
    try {
      await postFeedback(analysisId, { agrees, correctRhythm: rhythm, comment, clinician });
      try {
        localStorage.setItem(CLINICIAN_KEY, clinician);
      } catch {
        /* storage unavailable: ignore */
      }
      setSaved(true);
      setAgrees(null);
      setRhythm("");
      setComment("");
      onSaved?.();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  if (saved) {
    return (
      <div className="alert alert-green" role="status">
        <strong>Feedback saved. Thank you.</strong>
        <button type="button" className="btn btn-small btn-ghost" onClick={() => setSaved(false)}>
          Add another response
        </button>
      </div>
    );
  }

  return (
    <form onSubmit={submit} className="feedback-form">
      <p className="muted">Do you agree with the main rhythm result above?</p>
      <div className="segmented">
        <label className={agrees === true ? "on" : ""}>
          <input type="radio" name="agrees" checked={agrees === true} onChange={() => setAgrees(true)} />
          Agree
        </label>
        <label className={agrees === false ? "on on-bad" : ""}>
          <input type="radio" name="agrees" checked={agrees === false} onChange={() => setAgrees(false)} />
          Disagree
        </label>
      </div>

      {agrees === false && (
        <label className="field">
          <span>Correct rhythm (optional)</span>
          <select value={rhythm} onChange={(e) => setRhythm(e.target.value)}>
            <option value="">Not specified</option>
            {labels.map((l) => (
              <option key={l.id} value={l.id}>
                {l.name}
              </option>
            ))}
          </select>
        </label>
      )}

      <label className="field">
        <span>Comment (optional)</span>
        <textarea rows={2} value={comment} onChange={(e) => setComment(e.target.value)} placeholder="e.g. Flutter waves visible in II" />
      </label>
      <label className="field">
        <span>Your name or initials (optional)</span>
        <input type="text" value={clinician} onChange={(e) => setClinician(e.target.value)} placeholder="e.g. Dr. A" />
      </label>

      {error && <p className="field-error" role="alert">{error}</p>}
      <button type="submit" className="btn" disabled={busy}>
        {busy ? "Saving…" : "Submit feedback"}
      </button>
    </form>
  );
}
