import { Link, useParams } from "react-router-dom";
import { getAnalysis } from "../api.js";
import FeedbackForm from "../components/FeedbackForm.jsx";
import FeedbackList from "../components/FeedbackList.jsx";
import ResultView from "../components/ResultView.jsx";
import { Card, ErrorBox, Spinner } from "../components/ui.jsx";
import { useFetch } from "../hooks.js";
import { fmtLocal } from "../util.js";

export default function AnalysisDetail() {
  const { id } = useParams();
  const { loading, data, error, reload } = useFetch(() => getAnalysis(id), [id]);

  return (
    <div className="stack">
      <div className="row-between">
        <div>
          <Link to="/history">← History</Link>
          {data && (
            <small className="muted block">
              Analysis #{data.id} · {fmtLocal(data.created_at)}
            </small>
          )}
        </div>
        <Link to="/" className="btn btn-ghost btn-small">
          Analyse another
        </Link>
      </div>

      {loading && <Spinner label="Loading analysis…" />}
      {error && (
        <ErrorBox
          title={error.status === 404 ? "Analysis not found" : "Could not load this analysis"}
          message={error.message}
          onRetry={error.status === 404 ? undefined : reload}
        />
      )}

      {data && (
        <>
          <ResultView result={data.result} context={data.context} filename={data.filename} isTest={data.is_test} />
          <Card title="Feedback so far">
            <FeedbackList entries={data.feedback} />
          </Card>
          <Card title="Add feedback">
            <FeedbackForm analysisId={data.id} onSaved={reload} />
          </Card>
        </>
      )}
    </div>
  );
}
