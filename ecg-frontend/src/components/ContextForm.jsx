import { useMeta } from "../state.jsx";
import { DEFAULT_SYMPTOMS, symptomLabel, unstableReasons } from "../util.js";

/** Controlled patient-context form. `errors` is {age, systolicBp} -> message. */
export default function ContextForm({ value, onChange, errors = {} }) {
  const { symptoms } = useMeta();
  const symptomList = symptoms ?? DEFAULT_SYMPTOMS;
  const set = (patch) => onChange({ ...value, ...patch });
  const reasons = unstableReasons(value);

  const toggleSymptom = (s) =>
    set({ symptoms: value.symptoms.includes(s) ? value.symptoms.filter((x) => x !== s) : [...value.symptoms, s] });

  return (
    <div className="form-grid">
      <label className="field">
        <span>Age (years)</span>
        <input
          type="number"
          inputMode="numeric"
          min="0"
          max="130"
          step="1"
          value={value.age}
          onChange={(e) => set({ age: e.target.value })}
          placeholder="Not given"
        />
        {errors.age && <em className="field-error">{errors.age}</em>}
      </label>

      <label className="field">
        <span>Sex</span>
        <select value={value.sex} onChange={(e) => set({ sex: e.target.value })}>
          <option value="">Not given</option>
          <option value="male">Male</option>
          <option value="female">Female</option>
        </select>
      </label>

      <label className="field">
        <span>Systolic BP (mmHg)</span>
        <input
          type="number"
          inputMode="numeric"
          min="20"
          max="300"
          step="1"
          value={value.systolicBp}
          onChange={(e) => set({ systolicBp: e.target.value })}
          placeholder="Not given"
        />
        {errors.systolicBp && <em className="field-error">{errors.systolicBp}</em>}
      </label>

      <fieldset className="field field-wide">
        <legend>Is the patient haemodynamically stable?</legend>
        <div className="segmented">
          {[
            ["", "Not assessed"],
            ["true", "Yes, stable"],
            ["false", "No, unstable"],
          ].map(([v, label]) => (
            <label key={v || "na"} className={value.stable === v ? "on" : ""}>
              <input type="radio" name="stable" checked={value.stable === v} onChange={() => set({ stable: v })} />
              {label}
            </label>
          ))}
        </div>
      </fieldset>

      <fieldset className="field field-wide">
        <legend>Symptoms</legend>
        <div className="check-grid">
          {symptomList.map((s) => (
            <label key={s} className="check">
              <input type="checkbox" checked={value.symptoms.includes(s)} onChange={() => toggleSymptom(s)} />
              {symptomLabel(s)}
            </label>
          ))}
        </div>
      </fieldset>

      {reasons.length > 0 && (
        <div className="alert alert-amber field-wide" role="status">
          <strong>This patient will be treated as unstable</strong>, so emergency (ACLS) recommendations apply.
          <ul>
            {reasons.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
