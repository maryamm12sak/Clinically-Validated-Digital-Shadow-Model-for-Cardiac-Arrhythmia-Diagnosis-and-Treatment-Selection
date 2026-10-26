"""
Turns rhythm probabilities + measurements + patient context into
triage colour, findings and next steps, using rules.yaml.
No AI here: every recommendation comes word for word from rules.yaml.
"""
from pathlib import Path

import yaml

from labels import LABEL_NAMES

RULES = yaml.safe_load((Path(__file__).parent / "rules.yaml").read_text(encoding="utf-8"))
SEVERITY = {"green": 0, "amber": 1, "red": 2}


def _worst(*colours):
    return max(colours, key=lambda c: SEVERITY[c])


def assess_stability(context):
    """Returns (is_stable, list of reasons it is unstable)."""
    reasons = []
    if context.get("stable") is False:
        reasons.append("Marked unstable by clinician")
    sbp = context.get("systolic_bp")
    if sbp is not None and sbp < RULES["unstable_sbp_below"]:
        reasons.append(f"Systolic BP {sbp} mmHg (below {RULES['unstable_sbp_below']})")
    for s in context.get("symptoms", []):
        if s in RULES["unstable_symptoms"]:
            reasons.append(s.replace("_", " ").capitalize())
    return (len(reasons) == 0), reasons


def _apply_rule(label_id, stable, measurements, symptoms):
    rule = RULES["rhythms"][label_id]
    entry = rule["stable"] if stable else rule.get("unstable", rule["stable"])
    triage, steps = entry["triage"], list(entry["next_steps"])

    hr = measurements.get("heart_rate")
    for m in rule.get("modifiers", []):
        hit = (("if_symptom_any" in m and any(s in symptoms for s in m["if_symptom_any"])) or
               ("if_hr_above" in m and hr is not None and hr > m["if_hr_above"]) or
               ("if_hr_below" in m and hr is not None and hr < m["if_hr_below"]))
        if hit:
            triage = _worst(triage, m["triage"])
            steps += m["add_steps"]
    return triage, steps


def detected_rhythms(probabilities):
    thr = RULES["detection_threshold"]
    return [lid for lid, p in probabilities.items() if p >= thr]


def adjust_measurements(measurements, detected_ids):
    """
    Hide or flag measurements that don't make sense for the detected rhythm.
    Returns (adjusted measurements, list of notes explaining what was changed).
    """
    m, notes = dict(measurements), []
    d = set(detected_ids)
    if d & {"AFIB", "AFLT"}:
        m["pr_ms"] = None
        notes.append("PR not reported: there are no organised P waves in atrial fibrillation/flutter.")
    if "AVB_HIGH" in d:
        m["pr_ms"] = None
        notes.append("PR not reported: in high-grade AV block, P waves and QRS complexes are not consistently linked.")
    if "AFLT" in d:
        m["qt_ms"] = m["qtc_ms"] = None
        notes.append("QT/QTc not reported: flutter waves make the end of the T wave unmeasurable automatically.")
    elif "AFIB" in d and m.get("qtc_ms") is not None:
        notes.append("QT/QTc are approximate because the rhythm is irregular.")
    return m, notes


def _measurement_warnings(measurements, detected_ids):
    w = RULES["warnings"]
    out = []   # list of (triage, message)
    qtc, hr, qrs = measurements.get("qtc_ms"), measurements.get("heart_rate"), measurements.get("qrs_ms")
    bbb = any(d in ("LBBB", "RBBB") for d in detected_ids)

    wide = bbb or (qrs or 0) >= w["qrs_wide_ms"]
    if qtc is not None and qtc >= w["qtc_borderline_ms"] and wide:
        # a wide QRS inflates QTc, so don't escalate on it; flag for manual assessment instead
        out.append(("green", f"QTc {qtc} ms reads long, but the wide QRS inflates it. Assess repolarisation with a QRS-adjusted method (e.g. JT interval) before acting on it."))
    elif qtc is not None and qtc >= w["qtc_high_ms"]:
        out.append(("amber", f"QTc {qtc} ms (500 or more): high risk of torsades de pointes. Avoid QT-prolonging drugs; check and correct potassium and magnesium."))
    elif qtc is not None and qtc >= w["qtc_borderline_ms"]:
        out.append(("green", f"QTc {qtc} ms is prolonged: use QT-prolonging drugs with caution."))
    if hr is not None and hr < w["hr_low_bpm"]:
        out.append(("amber", f"Heart rate {hr} bpm is very low."))
    if hr is not None and hr > w["hr_high_bpm"]:
        out.append(("amber", f"Heart rate {hr} bpm is very high."))
    if qrs is not None and qrs >= w["qrs_wide_ms"] and not bbb:
        out.append(("green", f"QRS {qrs} ms is wide without a detected bundle branch block: review the ECG."))
    return out


def assess(probabilities, measurements, context):
    thr = RULES["detection_threshold"]
    symptoms = context.get("symptoms", [])
    stable, reasons = assess_stability(context)

    detected = sorted([(lid, p) for lid, p in probabilities.items() if p >= thr],
                      key=lambda x: x[1], reverse=True)
    if not detected:
        detected = [("UNCERTAIN", max(probabilities.values()) if probabilities else 0.0)]

    findings = []
    overall = "green"
    for lid, p in detected:
        triage, steps = _apply_rule(lid, stable, measurements, symptoms)
        overall = _worst(overall, triage)
        findings.append({"id": lid, "name": LABEL_NAMES[lid], "probability": round(p, 3),
                         "triage": triage, "next_steps": steps})

    warnings = []
    for triage, msg in _measurement_warnings(measurements, [d[0] for d in detected]):
        overall = _worst(overall, triage)
        warnings.append({"triage": triage, "message": msg})

    return {
        "triage": overall,
        "primary": findings[0]["id"],
        "stability": {"stable": stable, "reasons": reasons},
        "findings": findings,
        "warnings": warnings,
        "rules_version": RULES["version"],
        "disclaimer": RULES["disclaimer"],
    }
