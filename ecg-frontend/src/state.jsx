import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { getHealth, getMeta } from "./api.js";
import { DISCLAIMER_FALLBACK } from "./util.js";

// ---------- /meta + /health ----------

const MetaCtx = createContext(null);

export function MetaProvider({ children }) {
  const [meta, setMeta] = useState(null);
  const [metaError, setMetaError] = useState(null);
  const [health, setHealth] = useState({ state: "checking" }); // checking | online | offline

  const loadMeta = useCallback(() => {
    setMetaError(null);
    getMeta().then(setMeta).catch((e) => setMetaError(e.message));
  }, []);

  useEffect(() => {
    loadMeta();
  }, [loadMeta]);

  useEffect(() => {
    let alive = true;
    const ping = () =>
      getHealth()
        .then((h) => alive && setHealth({ state: "online", ...h }))
        .catch(() => alive && setHealth({ state: "offline" }));
    ping();
    const t = setInterval(ping, 20000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  // If the server came back after being offline at load time, fetch /meta again.
  useEffect(() => {
    if (health.state === "online" && !meta) loadMeta();
  }, [health.state, meta, loadMeta]);

  const value = useMemo(() => {
    const labels = meta?.labels ?? [];
    const names = Object.fromEntries(labels.map((l) => [l.id, l.name]));
    return {
      meta,
      metaError,
      health,
      labels,
      symptoms: meta?.symptoms ?? null,
      threshold: meta?.detection_threshold ?? 0.5,
      disclaimer: meta?.disclaimer ?? DISCLAIMER_FALLBACK,
      labelName: (id) => (id === "UNCERTAIN" ? "Uncertain – needs expert review" : (names[id] ?? id ?? "—")),
      reloadMeta: loadMeta,
    };
  }, [meta, metaError, health, loadMeta]);

  return <MetaCtx.Provider value={value}>{children}</MetaCtx.Provider>;
}

export const useMeta = () => useContext(MetaCtx);

// ---------- latest analysis (holds the big /analyze response incl. ECG signal) ----------

const LatestCtx = createContext(null);

export function LatestProvider({ children }) {
  const [latest, setLatest] = useState(null); // {id, result, testRhythm}
  const value = useMemo(() => ({ latest, setLatest }), [latest]);
  return <LatestCtx.Provider value={value}>{children}</LatestCtx.Provider>;
}

export const useLatest = () => useContext(LatestCtx);
