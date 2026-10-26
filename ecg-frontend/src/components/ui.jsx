const TRIAGE_TEXT = { green: "Routine", amber: "Needs attention", red: "Urgent" };

export function TriageBadge({ level, size = "md" }) {
  const known = level in TRIAGE_TEXT;
  return (
    <span className={`triage triage-${known ? level : "unknown"} triage-${size}`}>
      <span className="triage-dot" aria-hidden="true" />
      {known ? TRIAGE_TEXT[level] : (level ?? "Unknown")}
    </span>
  );
}

export function Spinner({ label = "Loading…" }) {
  return (
    <div className="spinner-row" role="status">
      <span className="spinner" aria-hidden="true" />
      {label}
    </div>
  );
}

export function ErrorBox({ message, onRetry, title = "Something went wrong" }) {
  return (
    <div className="alert alert-red" role="alert">
      <strong>{title}</strong>
      <div>{message}</div>
      {onRetry && (
        <button type="button" className="btn btn-small btn-ghost" onClick={onRetry}>
          Try again
        </button>
      )}
    </div>
  );
}

export function Card({ title, aside, children, className = "" }) {
  return (
    <section className={`card ${className}`}>
      {(title || aside) && (
        <header className="card-head">
          {title && <h2>{title}</h2>}
          {aside}
        </header>
      )}
      {children}
    </section>
  );
}

/** A small flat ECG trace for the sample cards (lead II, downsampled). */
export function Sparkline({ values, width = 300, height = 70 }) {
  if (!values?.length) return <div className="spark spark-empty" style={{ height }} />;
  const n = 300;
  const step = Math.max(1, Math.floor(values.length / n));
  const pts = [];
  for (let i = 0; i < values.length; i += step) pts.push(values[i]);
  const sorted = [...pts].sort((a, b) => a - b);
  const base = sorted[Math.floor(sorted.length / 2)];
  const half = Math.max(0.5, ...pts.map((v) => Math.abs(v - base))) * 1.1;
  const d = pts
    .map((v, i) => {
      const x = (i / (pts.length - 1)) * width;
      const y = height / 2 - ((v - base) / half) * (height / 2 - 4);
      return `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`;
    })
    .join(" ");
  return (
    <svg className="spark" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img" aria-label="ECG preview">
      <path d={d} fill="none" stroke="#111" strokeWidth="1.2" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}
