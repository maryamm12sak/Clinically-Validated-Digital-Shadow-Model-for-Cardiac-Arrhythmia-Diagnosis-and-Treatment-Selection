// One place for every call to the backend.
// All request bodies are FORM FIELDS (not JSON). Never set Content-Type by hand on FormData.

export const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message, status = 0, kind = "http") {
    super(message);
    this.status = status;
    this.kind = kind; // "network" | "http"
  }
}

const GENERIC_500 = "Something went wrong on the server. Please try again.";
export const ANALYZE_500 =
  "Could not analyse this ECG. The signal may have no detectable heartbeats or be unusual. Try another recording.";

function formatDetail(detail) {
  if (typeof detail === "string") return detail; // 400 / 404
  if (Array.isArray(detail)) {
    // 422: FastAPI validation list
    return detail
      .map((d) => {
        const loc = Array.isArray(d.loc) ? d.loc.filter((x) => !["body", "query", "path"].includes(x)).join(".") : "";
        return loc ? `${loc}: ${d.msg}` : d.msg;
      })
      .join("; ");
  }
  return null;
}

async function request(path, { method = "GET", body, serverError = GENERIC_500, signal } = {}) {
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, { method, body, signal });
  } catch (e) {
    if (e?.name === "AbortError") throw e;
    throw new ApiError(`Cannot reach the server at ${API_BASE}. Is the backend running?`, 0, "network");
  }
  if (res.ok) return res;

  let detail = null;
  try {
    detail = formatDetail((await res.json())?.detail);
  } catch {
    /* body was not JSON */
  }
  if (res.status >= 500) throw new ApiError(serverError, res.status);
  throw new ApiError(detail || `Request failed (${res.status}).`, res.status);
}

const json = async (path, opts) => (await request(path, opts)).json();

// ---------- files ----------

const EXT = (f) => f.name.split(".").pop().toLowerCase();

/** Works out which selected files are the ECG (.csv/.txt or .hea) and which is the WFDB .dat. */
export function classifyFiles(files) {
  const list = [...files];
  if (!list.length) return { main: null, dat: null, error: null };
  const hea = list.filter((f) => EXT(f) === "hea");
  const dat = list.filter((f) => EXT(f) === "dat");
  const tab = list.filter((f) => ["csv", "txt"].includes(EXT(f)));
  const other = list.filter((f) => !["hea", "dat", "csv", "txt"].includes(EXT(f)));

  if (other.length) {
    const e = EXT(other[0]);
    const msg = ["xml"].includes(e)
      ? "XML files aren't supported yet."
      : ["png", "jpg", "jpeg", "pdf"].includes(e)
        ? "Photos or scans of ECGs aren't supported yet."
        : `Unsupported file type: ${other[0].name}.`;
    return { main: null, dat: null, error: `${msg} Upload a 12-lead CSV, or a WFDB .hea + .dat pair.` };
  }
  if (tab.length) {
    if (tab.length > 1 || hea.length || dat.length)
      return { main: null, dat: null, error: "Choose either one CSV file or one .hea + .dat pair, not both." };
    return { main: tab[0], dat: null, error: null };
  }
  if (hea.length > 1 || dat.length > 1)
    return { main: null, dat: null, error: "Choose just one .hea file and one .dat file." };
  if (hea.length && dat.length) return { main: hea[0], dat: dat[0], error: null };
  return {
    main: null,
    dat: null,
    error: hea.length ? "Now add the matching .dat file." : "Now add the matching .hea file.",
    incomplete: true,
  };
}

function fileForm({ main, dat }, fs) {
  const form = new FormData();
  form.append("file", main, main.name);
  if (dat) form.append("file2", dat, dat.name);
  if (fs !== undefined && fs !== null && String(fs).trim() !== "") form.append("fs", String(fs).trim());
  return form;
}

// ---------- endpoints ----------

export const getHealth = (signal) => json("/health", { signal });
export const getMeta = () => json("/meta");
export const getFeedbackSummary = () => json("/feedback/summary");

export const parseEcg = (files, fs, signal) =>
  json("/parse", { method: "POST", body: fileForm(files, fs), signal });

export function analyze({ files, fs, context, testRhythm }) {
  const form = fileForm(files, fs);
  // Blank = "not given" for the backend, so it's safe to always send these.
  form.append("age", context.age ?? "");
  form.append("sex", context.sex ?? "");
  form.append("systolic_bp", context.systolicBp ?? "");
  form.append("symptoms", (context.symptoms ?? []).join(","));
  form.append("stable", context.stable ?? "");
  if (testRhythm) form.append("test_rhythm", testRhythm);
  return json("/analyze", { method: "POST", body: form, serverError: ANALYZE_500 });
}

export const listSamples = () => json("/samples");
export const getSample = (id, signal) => json(`/samples/${encodeURIComponent(id)}`, { signal });
export async function getSampleFile(id) {
  const res = await request(`/samples/${encodeURIComponent(id)}/file`);
  return res.blob();
}

export const listAnalyses = ({ limit = 100, includeTests = true } = {}) =>
  json(`/analyses?limit=${limit}&include_tests=${includeTests}`);
export const getAnalysis = (id) => json(`/analyses/${encodeURIComponent(id)}`);

export function postFeedback(analysisId, { agrees, correctRhythm, comment, clinician }) {
  const form = new FormData();
  form.append("agrees", agrees ? "true" : "false");
  if (!agrees && correctRhythm) form.append("correct_rhythm", correctRhythm);
  if (comment?.trim()) form.append("comment", comment.trim());
  if (clinician?.trim()) form.append("clinician", clinician.trim());
  return json(`/analyses/${encodeURIComponent(analysisId)}/feedback`, { method: "POST", body: form });
}
