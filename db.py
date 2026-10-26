"""
SQLite storage: analysis history (audit log) and doctor feedback.

Uses Python's built-in sqlite3, so nothing to install. The database is one file, ecg.db,
created automatically next to this file. To move to a cloud database later (e.g. PostgreSQL),
only this file needs to change.

Privacy: no ECG signal and no patient name/ID is stored, only the uploaded file name,
the clinical context entered (age, sex, BP, symptoms) and the tool's output.
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "ecg.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at     TEXT NOT NULL,
    filename       TEXT,
    context        TEXT NOT NULL,      -- JSON: age, sex, systolic_bp, symptoms, stable
    primary_rhythm TEXT,
    triage         TEXT,
    measurements   TEXT,               -- JSON
    result         TEXT NOT NULL,      -- JSON: full response minus the plot data
    model_version  TEXT,
    rules_version  TEXT,
    is_test        INTEGER NOT NULL DEFAULT 0   -- 1 if test_rhythm was used
);
CREATE TABLE IF NOT EXISTS feedback (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id    INTEGER NOT NULL REFERENCES analyses(id),
    created_at     TEXT NOT NULL,
    agrees         INTEGER NOT NULL,   -- 1 = doctor agrees with the tool, 0 = disagrees
    correct_rhythm TEXT,               -- the doctor's rhythm if they disagree
    comment        TEXT,
    clinician      TEXT
);
"""


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with _connect() as conn:
        conn.executescript(SCHEMA)


def save_analysis(filename, context, result, is_test):
    """Stores one analysis and returns its id."""
    stored = {k: v for k, v in result.items() if k != "ecg"}   # plot data is large and not needed
    a = result["assessment"]
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO analyses (created_at, filename, context, primary_rhythm, triage,
                                     measurements, result, model_version, rules_version, is_test)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (_now(), filename, json.dumps(context), a["primary"], a["triage"],
             json.dumps(result["measurements"]), json.dumps(stored),
             result["model"]["version"], a["rules_version"], int(is_test)))
        return cur.lastrowid


def _feedback_for(conn, analysis_id):
    rows = conn.execute("SELECT * FROM feedback WHERE analysis_id = ? ORDER BY id", (analysis_id,))
    return [{**dict(r), "agrees": bool(r["agrees"])} for r in rows]


def list_analyses(limit=50, include_tests=True):
    sql = "SELECT * FROM analyses" + ("" if include_tests else " WHERE is_test = 0") + " ORDER BY id DESC LIMIT ?"
    with _connect() as conn:
        out = []
        for r in conn.execute(sql, (limit,)):
            fb = _feedback_for(conn, r["id"])
            out.append({
                "id": r["id"], "created_at": r["created_at"], "filename": r["filename"],
                "primary_rhythm": r["primary_rhythm"], "triage": r["triage"],
                "measurements": json.loads(r["measurements"]), "context": json.loads(r["context"]),
                "is_test": bool(r["is_test"]),
                "feedback": fb[-1] if fb else None,
            })
        return out


def get_analysis(analysis_id):
    with _connect() as conn:
        r = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
        if r is None:
            return None
        return {"id": r["id"], "created_at": r["created_at"], "filename": r["filename"],
                "is_test": bool(r["is_test"]), "context": json.loads(r["context"]),
                "result": json.loads(r["result"]), "feedback": _feedback_for(conn, analysis_id)}


def add_feedback(analysis_id, agrees, correct_rhythm=None, comment=None, clinician=None):
    with _connect() as conn:
        if conn.execute("SELECT 1 FROM analyses WHERE id = ?", (analysis_id,)).fetchone() is None:
            return None
        cur = conn.execute(
            """INSERT INTO feedback (analysis_id, created_at, agrees, correct_rhythm, comment, clinician)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (analysis_id, _now(), int(agrees), correct_rhythm, comment, clinician))
        return cur.lastrowid


def feedback_summary():
    """Agreement between doctors and the tool, for the evaluation."""
    with _connect() as conn:
        row = conn.execute(
            """SELECT COUNT(*) AS n, SUM(f.agrees) AS agreed
               FROM feedback f JOIN analyses a ON a.id = f.analysis_id
               WHERE a.is_test = 0
                 AND f.id = (SELECT MAX(id) FROM feedback WHERE analysis_id = f.analysis_id)""").fetchone()
        n, agreed = row["n"], row["agreed"] or 0
        return {"reviewed": n, "agreed": agreed, "agreement_rate": round(agreed / n, 3) if n else None}
