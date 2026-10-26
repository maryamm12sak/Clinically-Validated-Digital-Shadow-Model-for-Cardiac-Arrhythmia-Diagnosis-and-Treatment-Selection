import { Link, Navigate, useParams } from "react-router-dom";
import FeedbackForm from "../components/FeedbackForm.jsx";
import ResultView from "../components/ResultView.jsx";
import { Card } from "../components/ui.jsx";
import { useLatest } from "../state.jsx";

export default function Results() {
  const { id } = useParams();
  const { latest } = useLatest();

  // After a page refresh the big in-memory response is gone: fall back to the stored record.
  if (!latest || String(latest.id) !== id) return <Navigate to={`/history/${id}`} replace />;

  const { result, testRhythm } = latest;
  return (
    <div className="stack">
      <div className="row-between">
        <small className="muted">Analysis #{result.analysis_id}</small>
        <div className="row">
          <Link to="/" className="btn btn-ghost btn-small">
            Analyse another
          </Link>
          <Link to={`/history/${id}`} className="btn btn-ghost btn-small">
            View in history
          </Link>
        </div>
      </div>

      <ResultView
        result={result}
        context={result.context}
        ecg={result.ecg}
        filename={result.filenames?.join(", ")}
        testRhythm={testRhythm}
      />

      <Card title="Your feedback">
        <FeedbackForm analysisId={result.analysis_id} />
      </Card>
    </div>
  );
}
