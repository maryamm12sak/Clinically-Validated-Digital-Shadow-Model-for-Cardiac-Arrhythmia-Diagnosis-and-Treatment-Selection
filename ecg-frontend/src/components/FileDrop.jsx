import { useRef, useState } from "react";

export default function FileDrop({ files, onFiles, error, hint }) {
  const input = useRef(null);
  const [over, setOver] = useState(false);

  const take = (list) => {
    if (list && list.length) onFiles([...list]);
  };

  return (
    <div>
      <div
        className={`dropzone ${over ? "over" : ""} ${error ? "has-error" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          take(e.dataTransfer.files);
        }}
        onClick={() => input.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
      >
        <input
          ref={input}
          type="file"
          multiple
          accept=".csv,.txt,.hea,.dat"
          hidden
          onChange={(e) => {
            take(e.target.files);
            e.target.value = ""; // allow re-selecting the same file
          }}
        />
        {files.length === 0 ? (
          <>
            <strong>Drop an ECG file here, or click to browse</strong>
            <span>12-lead CSV / TXT — or a WFDB pair (.hea + .dat, select both together)</span>
          </>
        ) : (
          <ul className="file-chips">
            {files.map((f) => (
              <li key={f.name}>
                {f.name} <small>{(f.size / 1024).toFixed(0)} KB</small>
              </li>
            ))}
          </ul>
        )}
      </div>
      {files.length > 0 && (
        <button type="button" className="btn btn-small btn-ghost" onClick={() => onFiles([])}>
          Remove file{files.length > 1 ? "s" : ""}
        </button>
      )}
      {error && <p className="field-error">{error}</p>}
      {!error && hint && <p className="hint">{hint}</p>}
    </div>
  );
}
