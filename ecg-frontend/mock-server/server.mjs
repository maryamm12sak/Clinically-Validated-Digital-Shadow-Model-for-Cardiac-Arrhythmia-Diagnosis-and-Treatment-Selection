// Tiny stand-in for the FastAPI backend so the frontend can be developed and demoed
// without it. Mirrors the real contract (field names, error shapes). NOT clinically meaningful.
//   npm run mock      -> http://127.0.0.1:8000
//
// Handy triggers: a file name containing "bad" returns a 400, "crash" returns a 500.
import http from "node:http";

const PORT = Number(process.env.PORT || 8000);
const LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"];
const LABELS = [
  ["NSR", "Normal sinus rhythm"], ["SBRAD", "Sinus bradycardia"], ["STACH", "Sinus tachycardia"],
  ["AFIB", "Atrial fibrillation"], ["AFLT", "Atrial flutter"], ["SVT", "Supraventricular tachycardia"],
  ["AVB1", "First-degree AV block"], ["AVB_HIGH", "Second/third-degree AV block"], ["LBBB", "Left bundle branch block"],
  ["RBBB", "Right bundle branch block"], ["PVC", "Premature ventricular complexes"], ["PAC", "Premature atrial complexes"],
];
const NAME = Object.fromEntries(LABELS);
const SYMPTOMS = ["chest_pain", "dyspnea", "palpitations", "syncope", "dizziness", "altered_mental_status", "shock", "heart_failure"];
const DISCLAIMER = "Decision support only. Draft recommendations pending clinical review. The treating physician is responsible for all decisions.";
const SAMPLES = [
  ["normal", "Normal ECG", 72], ["sinus_brady", "Sinus bradycardia", 48], ["sinus_tachy", "Sinus tachycardia", 118],
  ["afib", "Atrial fibrillation", 95], ["aflutter", "Atrial flutter", 150], ["avb1", "First-degree AV block", 64],
  ["avb3", "Third-degree AV block", 38], ["lbbb", "Left bundle branch block", 70], ["rbbb", "Right bundle branch block", 76],
  ["pvc", "Premature ventricular complexes", 80],
].map(([id, description, hr], i) => ({
  id, description, hr, file: `${id}.csv`, source: `PTB-XL ecg_id ${7000 + i * 131}`, ptbxl_codes: [id.toUpperCase()],
  fs: 100, age: id === "afib" ? "90+" : 45 + i * 4, sex: i % 2 ? "female" : "male",
}));

// ---- synthetic ECG ----
const g = (t, mu, s, a) => a * Math.exp(-((t - mu) ** 2) / (2 * s * s));
function leadValues(hr, k, irregular) {
  const fs = 100, n = 1000, out = [];
  const amp = [1, 1.3, 0.5, -0.9, 0.35, 0.9, 0.3, 0.8, 1.2, 1.4, 1.1, 0.8][k];
  let beats = [], t = 0.4, seed = 7;
  const rnd = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  while (t < 10.5) { beats.push(t); t += (60 / hr) * (irregular ? 0.65 + rnd() * 0.7 : 1); }
  for (let i = 0; i < n; i++) {
    const x = i / fs; let v = 0.02 * Math.sin(x * 1.3 + k);
    for (const b of beats) {
      const d = x - b;
      if (d < -0.4 || d > 0.7) continue;
      v += amp * (g(d, -0.16, 0.025, 0.12) + g(d, -0.03, 0.009, -0.12) + g(d, 0, 0.011, 1.0) + g(d, 0.035, 0.01, -0.2) + g(d, 0.25, 0.05, 0.28));
    }
    out.push(+v.toFixed(4));
  }
  return out;
}
const ecgFor = (hr, irregular = false) => ({
  fs: 100, duration_s: 10, units: "mV", leads: LEADS.map((name, k) => ({ name, values: leadValues(hr, k, irregular) })),
});
const csvFor = (hr) => {
  const cols = LEADS.map((_, k) => leadValues(hr, k, false));
  return LEADS.join(",") + "\n" + Array.from({ length: 1000 }, (_, i) => cols.map((c) => c[i]).join(",")).join("\n") + "\n";
};

// ---- fake rules ----
const BASE = {
  NSR: ["green", ["No arrhythmia detected. Routine follow-up."]],
  SBRAD: ["green", ["Check for symptoms and reversible causes (medications, hypothyroidism).", "Observe if asymptomatic."]],
  STACH: ["amber", ["Look for the underlying cause: fever, pain, anaemia, hypovolaemia, hyperthyroidism."]],
  AFIB: ["amber", ["Rate control (e.g. beta-blocker or diltiazem) if stable.", "Assess stroke risk and plan anticoagulation."]],
  AFLT: ["amber", ["Rate control if stable; consider early cardiology input.", "Plan anticoagulation as for AF."]],
  SVT: ["amber", ["Vagal manoeuvres first.", "Adenosine if vagal manoeuvres fail."]],
  AVB1: ["green", ["Review AV-nodal blocking drugs. Usually no urgent treatment."]],
  AVB_HIGH: ["red", ["Prepare for atropine / transcutaneous pacing.", "Urgent cardiology referral."]],
  LBBB: ["amber", ["If new, treat as possible ACS until proven otherwise."]],
  RBBB: ["green", ["Often benign; consider PE if acute dyspnea."]],
  PVC: ["green", ["Check electrolytes (K, Mg). Treat only if symptomatic or frequent."]],
  PAC: ["green", ["Usually benign. Reduce caffeine and stimulants."]],
};
const UNSTABLE_STEPS = {
  AFIB: ["Synchronized cardioversion (ACLS: 120-200 J biphasic for atrial fibrillation).", "Sedate before cardioversion if the patient is conscious and time allows.", "Start or plan anticoagulation around cardioversion."],
  AFLT: ["Synchronized cardioversion (ACLS: 50-100 J biphasic for atrial flutter)."],
  SVT: ["Synchronized cardioversion (ACLS: 50-100 J)."],
};
const RANK = { green: 0, amber: 1, red: 2 };

// ---- in-memory store ----
const analyses = [];
const feedback = [];
let aid = 0, fid = 0;

function analyse(fields, filename) {
  const test = (fields.test_rhythm || "").toUpperCase();
  if (test && !NAME[test]) return [400, { detail: `test_rhythm must be one of [${LABELS.map((l) => l[0]).join(", ")}]` }];
  const primary = test || "NSR";
  const rest = LABELS.map((l) => l[0]).filter((x) => x !== primary);
  const probs = { [primary]: test ? 0.95 : 0.91 };
  rest.forEach((id, i) => (probs[id] = i === 0 ? 0.03 : i === 1 ? 0.02 : 0.003));
  const rhythms = Object.entries(probs).sort((a, b) => b[1] - a[1]).map(([id, probability]) => ({ id, name: NAME[id], probability }));

  const symptoms = (fields.symptoms || "").split(",").map((s) => s.trim()).filter(Boolean);
  const bad = symptoms.filter((s) => !SYMPTOMS.includes(s));
  if (bad.length) return [400, { detail: `Unknown symptoms ${JSON.stringify(bad)}. Allowed: ${JSON.stringify(SYMPTOMS)}` }];
  const num = (k) => (fields[k] === undefined || fields[k] === "" ? null : Number(fields[k]));
  const age = num("age"), bp = num("systolic_bp");
  if ((age !== null && !Number.isInteger(age)) ) return [400, { detail: "age must be a number." }];
  if ((bp !== null && !Number.isInteger(bp))) return [400, { detail: "systolic_bp must be a number." }];
  const stableF = fields.stable === "true" ? true : fields.stable === "false" ? false : null;
  const reasons = [];
  if (stableF === false) reasons.push("Marked as not stable by the clinician");
  if (bp !== null && bp < 90) reasons.push(`Systolic BP ${bp} mmHg (below 90)`);
  for (const s of ["altered_mental_status", "shock", "heart_failure"]) if (symptoms.includes(s)) reasons.push(`Symptom: ${s.replace(/_/g, " ")}`);
  const unstable = reasons.length > 0;

  const sample = SAMPLES.find((s) => filename.startsWith(s.id));
  const hr = sample?.hr ?? { SBRAD: 48, STACH: 118, AFIB: 95, AFLT: 150, SVT: 170, AVB_HIGH: 38 }[primary] ?? 72;
  const noPR = ["AFIB", "AFLT", "AVB_HIGH"].includes(primary);
  const measurements = { heart_rate: hr, pr_ms: noPR ? null : primary === "AVB1" ? 230 : 160, qrs_ms: primary === "LBBB" ? 130 : primary === "RBBB" ? 140 : 88, qt_ms: primary === "AFLT" ? null : 380, qtc_ms: primary === "AFLT" ? null : primary === "AFIB" ? 512 : 410 };
  const notes = [];
  if (primary === "AFIB" || primary === "AFLT") notes.push("PR not reported: there are no organised P waves in atrial fibrillation/flutter.");
  if (primary === "AFIB") notes.push("QT/QTc are approximate because the rhythm is irregular.");
  if (primary === "AFLT") notes.push("QT/QTc not reported for atrial flutter.");

  const [baseTriage, baseSteps] = BASE[primary];
  const triage = unstable && ["AFIB", "AFLT", "SVT"].includes(primary) ? "red" : baseTriage;
  const finding = { id: primary, name: NAME[primary], probability: probs[primary], triage, next_steps: unstable && UNSTABLE_STEPS[primary] ? UNSTABLE_STEPS[primary] : baseSteps };
  const warnings = [];
  if (measurements.qtc_ms >= 500) warnings.push({ triage: "amber", message: `QTc ${measurements.qtc_ms} ms (500 or more): high risk of torsades de pointes. Avoid QT-prolonging drugs; check and correct potassium and magnesium.` });
  if (measurements.heart_rate < 50) warnings.push({ triage: "green", message: `Heart rate ${measurements.heart_rate} bpm is below 50.` });
  const overall = [triage, ...warnings.map((w) => w.triage)].reduce((a, b) => (RANK[b] > RANK[a] ? b : a));

  const result = {
    model: { loaded: false, version: null, note: "Placeholder rhythm probabilities: the trained model is not plugged in yet." },
    rhythms, measurements, measurement_notes: notes,
    assessment: { triage: overall, primary, stability: { stable: !unstable, reasons }, findings: [finding], warnings, rules_version: "draft-0.1", disclaimer: DISCLAIMER },
  };
  const context = { age, sex: fields.sex || null, systolic_bp: bp, symptoms, stable: stableF };
  const row = { id: ++aid, created_at: new Date().toISOString().replace(/\.\d+Z$/, "+00:00"), filename, context, primary_rhythm: primary, triage: overall, measurements, result, is_test: Boolean(test) };
  analyses.push(row);
  return [200, { analysis_id: row.id, filenames: [filename], context, ...result, ecg: ecgFor(hr, primary === "AFIB") }];
}

// ---- multipart / urlencoded parsing (just enough) ----
function parseBody(buf, contentType = "") {
  const fields = {}, files = [];
  const m = /boundary=(?:"([^"]+)"|([^;]+))/.exec(contentType);
  if (m) {
    const sep = Buffer.from("--" + (m[1] || m[2]));
    let pos = buf.indexOf(sep);
    while (pos !== -1) {
      const next = buf.indexOf(sep, pos + sep.length);
      if (next === -1) break;
      const part = buf.subarray(pos + sep.length + 2, next - 2);
      const headEnd = part.indexOf("\r\n\r\n");
      const head = part.subarray(0, headEnd).toString();
      const body = part.subarray(headEnd + 4);
      const name = /name="([^"]*)"/.exec(head)?.[1];
      const filename = /filename="([^"]*)"/.exec(head)?.[1];
      if (filename !== undefined) files.push({ name, filename, size: body.length });
      else if (name) fields[name] = body.toString();
      pos = next;
    }
  } else if (contentType.includes("urlencoded")) {
    for (const [k, v] of new URLSearchParams(buf.toString())) fields[k] = v;
  }
  return { fields, files };
}

// ---- server ----
const send = (res, status, body, type = "application/json") => {
  res.writeHead(status, { "Content-Type": type, "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "*", "Access-Control-Allow-Methods": "*" });
  res.end(type === "application/json" ? JSON.stringify(body) : body);
};

http.createServer(async (req, res) => {
  if (req.method === "OPTIONS") return send(res, 204, "", "text/plain");
  const url = new URL(req.url, "http://x");
  const p = url.pathname;
  const chunks = [];
  for await (const c of req) chunks.push(c);
  const { fields, files } = parseBody(Buffer.concat(chunks), req.headers["content-type"]);
  console.log(req.method, p + url.search);

  if (p === "/health") return send(res, 200, { status: "ok", model_loaded: false, model_version: null });
  if (p === "/meta")
    return send(res, 200, { labels: LABELS.map(([id, name]) => ({ id, name })), model_version: null, metrics: null, rules_version: "draft-0.1", detection_threshold: 0.5, symptoms: SYMPTOMS, disclaimer: DISCLAIMER });
  if (p === "/samples") return send(res, 200, { samples: SAMPLES.map(({ hr, ...s }) => s) });

  let m = /^\/samples\/([^/]+)(\/file)?$/.exec(p);
  if (m) {
    const s = SAMPLES.find((x) => x.id === m[1]);
    if (!s) return send(res, 404, { detail: `No sample called '${m[1]}'.` });
    if (m[2]) return send(res, 200, csvFor(s.hr), "text/csv");
    const { hr, ...rest } = s;
    return send(res, 200, { ...rest, ecg: ecgFor(hr, s.id === "afib") });
  }

  if (p === "/parse" || p === "/analyze") {
    const f = files.find((x) => x.name === "file");
    if (!f) return send(res, 422, { detail: [{ type: "missing", loc: ["body", "file"], msg: "Field required", input: null }] });
    const lower = f.filename.toLowerCase();
    if (lower.includes("crash")) return send(res, 500, { detail: "Internal Server Error" });
    if (lower.includes("bad")) return send(res, 400, { detail: "Expected 12 leads, found 8." });
    if (!/\.(csv|txt|hea)$/.test(lower)) return send(res, 400, { detail: "Unsupported file type. Upload a 12-lead CSV, or a WFDB .hea + .dat pair." });
    if (p === "/parse") return send(res, 200, { filenames: [f.filename], ecg: ecgFor(72) });
    const [status, body] = analyse(fields, f.filename);
    return send(res, status, body);
  }

  if (p === "/analyses" && req.method === "GET") {
    const includeTests = url.searchParams.get("include_tests") !== "false";
    const limit = Math.min(500, Math.max(1, Number(url.searchParams.get("limit") || 50)));
    const rows = [...analyses].reverse().filter((a) => includeTests || !a.is_test).slice(0, limit).map((a) => ({
      id: a.id, created_at: a.created_at, filename: a.filename, primary_rhythm: a.primary_rhythm, triage: a.triage,
      measurements: a.measurements, context: a.context, is_test: a.is_test, feedback: [...feedback].reverse().find((f) => f.analysis_id === a.id) ?? null,
    }));
    return send(res, 200, { analyses: rows });
  }
  m = /^\/analyses\/(\d+)$/.exec(p);
  if (m) {
    const a = analyses.find((x) => x.id === Number(m[1]));
    if (!a) return send(res, 404, { detail: `No analysis with id ${m[1]}.` });
    return send(res, 200, { id: a.id, created_at: a.created_at, filename: a.filename, is_test: a.is_test, context: a.context, result: a.result, feedback: feedback.filter((f) => f.analysis_id === a.id) });
  }
  if (/^\/analyses\/[^/]+$/.test(p)) return send(res, 422, { detail: [{ type: "int_parsing", loc: ["path", "analysis_id"], msg: "Input should be a valid integer", input: p.split("/")[2] }] });

  m = /^\/analyses\/(\d+)\/feedback$/.exec(p);
  if (m && req.method === "POST") {
    const a = analyses.find((x) => x.id === Number(m[1]));
    if (!a) return send(res, 404, { detail: `No analysis with id ${m[1]}.` });
    if (fields.agrees === undefined) return send(res, 422, { detail: [{ type: "missing", loc: ["body", "agrees"], msg: "Field required", input: null }] });
    const v = fields.agrees.toLowerCase();
    if (!["true", "false", "yes", "no", "1", "0"].includes(v)) return send(res, 400, { detail: "agrees must be true or false." });
    const cr = (fields.correct_rhythm || "").toUpperCase();
    if (cr && !NAME[cr]) return send(res, 400, { detail: `correct_rhythm must be one of [${LABELS.map((l) => l[0]).join(", ")}]` });
    const row = { id: ++fid, analysis_id: a.id, created_at: new Date().toISOString().replace(/\.\d+Z$/, "+00:00"), agrees: ["true", "yes", "1"].includes(v), correct_rhythm: cr || null, comment: fields.comment || null, clinician: fields.clinician || null };
    feedback.push(row);
    return send(res, 200, { feedback_id: row.id, analysis_id: a.id });
  }

  if (p === "/feedback/summary") {
    const real = analyses.filter((a) => !a.is_test);
    const latest = real.map((a) => [...feedback].reverse().find((f) => f.analysis_id === a.id)).filter(Boolean);
    const agreed = latest.filter((f) => f.agrees).length;
    return send(res, 200, { reviewed: latest.length, agreed, agreement_rate: latest.length ? agreed / latest.length : null });
  }

  send(res, 404, { detail: "Not Found" });
}).listen(PORT, "127.0.0.1", () => console.log(`Mock ECG API on http://127.0.0.1:${PORT}`));
