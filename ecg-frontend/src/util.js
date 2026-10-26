export const DISCLAIMER_FALLBACK =
  "Decision support only. Draft recommendations pending clinical review. The treating physician is responsible for all decisions.";

export const SYMPTOM_LABELS = {
  chest_pain: "Chest pain",
  dyspnea: "Shortness of breath (dyspnea)",
  palpitations: "Palpitations",
  syncope: "Syncope (fainting)",
  dizziness: "Dizziness",
  altered_mental_status: "Altered mental status",
  shock: "Shock",
  heart_failure: "Heart failure",
};
export const DEFAULT_SYMPTOMS = Object.keys(SYMPTOM_LABELS);
export const UNSTABLE_SYMPTOMS = ["altered_mental_status", "shock", "heart_failure"];

export const symptomLabel = (id) =>
  SYMPTOM_LABELS[id] ?? id.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());

// Demo sample id -> the rhythm label it should show (used only by the test-mode toggle).
export const SAMPLE_LABEL = {
  normal: "NSR",
  sinus_brady: "SBRAD",
  sinus_tachy: "STACH",
  afib: "AFIB",
  aflutter: "AFLT",
  avb1: "AVB1",
  avb3: "AVB_HIGH",
  lbbb: "LBBB",
  rbbb: "RBBB",
  pvc: "PVC",
};

export const SHOW_TEST_TOOLS = import.meta.env.VITE_SHOW_TEST_TOOLS === "true";

export function fmtLocal(iso) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleString("en-GB", {
      timeZone: "Asia/Karachi",
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
  } catch {
    return iso;
  }
}

export const pct = (p) => (typeof p === "number" ? `${Math.round(p * 100)}%` : "—");

/** Client-side mirror of the backend stability rule, used only for a live hint in the form. */
export function unstableReasons(ctx) {
  const reasons = [];
  if (ctx.stable === "false") reasons.push("Marked as not stable");
  if (ctx.systolicBp !== "" && Number(ctx.systolicBp) < 90)
    reasons.push(`Systolic BP ${ctx.systolicBp} mmHg (below 90)`);
  for (const s of ctx.symptoms) {
    if (UNSTABLE_SYMPTOMS.includes(s)) reasons.push(`Symptom: ${symptomLabel(s).toLowerCase()}`);
  }
  return reasons;
}

export const EMPTY_CONTEXT = { age: "", sex: "", systolicBp: "", symptoms: [], stable: "" };
