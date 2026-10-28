"""
Fold-9 ensembles, calibration and thresholds from saved predictions (no training, fold 10 untouched).

    python analyze_round2.py

Writes ml/results/ensembles/ and ml/results/calibration/ and prints a summary.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ML_DIR))

from src import calibration as cal  # noqa: E402
from src.data_loader import PTBXL, load_config  # noqa: E402
from src.ensemble import (apply_stacker, apply_weights, cross_validated, fit_stacker,  # noqa: E402
                          fit_weights, load_model, summarize)
from src.labels import label_ids, load_label_config  # noqa: E402

RESULTS = ML_DIR / "results"
PRED = RESULTS / "predictions"
SEEDS = [0, 1, 2]
BASE = {  # name -> run_id prefix
    "lightgbm": "lightgbm__signal",
    "resnet1d": "resnet1d__raw100",
    "simple_cnn": "simple_cnn__raw100",
    "cnn_lstm": "cnn_lstm__raw100",
    "inception1d": "inception1d__raw100",
}
DEEP_NAMES = ["resnet1d", "simple_cnn", "cnn_lstm", "inception1d"]
# CPU cost of LightGBM's 500 Hz feature pipeline: 165 s wall on 16 workers for 21,799 records
FEATURE_MS_PER_RECORD = 165.3 * 16 * 1000 / 21799


def main():
    config = load_config()
    labels = load_label_config(ML_DIR / config["data"]["labels_config"])
    ids = label_ids(labels)
    tiers = [lab["tier"] for lab in labels]
    db = PTBXL(config, sampling_rate=100).db
    bench = pd.read_csv(RESULTS / "benchmark_results.csv")
    bench = bench[bench["smoke"].astype(str) == "False"]

    avail, seeds_of, Y, ecg = {}, {}, None, None
    for name, prefix in BASE.items():
        have = [s for s in SEEDS if (PRED / f"{prefix}__seed{s}.npz").exists()]
        if not have:
            continue
        runs, y, e = load_model(PRED, prefix, have)
        if Y is None:
            Y, ecg = y, e
        assert np.array_equal(e, ecg) and np.array_equal(y, Y)
        avail[name], seeds_of[name] = runs, have
    groups = db.loc[ecg, "patient_id"].values
    print("models:", {k: seeds_of[k] for k in avail})

    infer_ms = {name: float(bench[bench["run_id"].str.startswith(BASE[name] + "__seed")]
                            ["inference_ms_per_record"].mean()) for name in avail}
    infer_ms["lightgbm"] = infer_ms.get("lightgbm", 0) + FEATURE_MS_PER_RECORD

    rows = []
    # single models: per-seed mean +- SD, and the 3-seed average
    single = {}
    for name, runs in avail.items():
        per_seed = [summarize(Y, P, labels, groups, n_boot=0)[1] for P in runs]
        au = [s["macro_auroc_primary"] for s in per_seed]
        ap = [s["macro_auprc_primary"] for s in per_seed]
        single[name] = (np.mean(au), np.std(au, ddof=1) if len(au) > 1 else np.nan)
        rows.append({"candidate": name, "method": "single model", "seeds": "per-seed mean+-SD",
                     "macro_auroc": np.mean(au), "auroc_sd": single[name][1],
                     "macro_auprc": np.mean(ap), "auprc_sd": np.std(ap, ddof=1) if len(ap) > 1 else np.nan,
                     "n_seeds": len(runs), "cpu_ms_per_record": infer_ms[name]})
        _, s = summarize(Y, np.mean(runs, axis=0), labels, groups)
        rows.append({"candidate": name, "method": "seed average", "seeds": "seed-averaged",
                     "macro_auroc": s["macro_auroc_primary"], "ci_low": s["ci_low"],
                     "ci_high": s["ci_high"], "macro_auprc": s["macro_auprc_primary"],
                     "n_seeds": len(runs), "cpu_ms_per_record": infer_ms[name] * len(runs)})
    best_single = max(single, key=lambda k: single[k][0])
    best_auroc, best_sd = single[best_single]

    candidates = {"resnet1d+lightgbm": ["resnet1d", "lightgbm"],
                  "resnet1d+inception1d": ["resnet1d", "inception1d"],
                  "all_deep": [n for n in DEEP_NAMES],
                  "all_deep+lightgbm": DEEP_NAMES + ["lightgbm"]}
    ens_preds = {}
    for cname, members in candidates.items():
        members = [m for m in members if m in avail]
        if len(members) < 2:
            print(f"skip {cname}: members available {members}")
            continue
        n_seed = min(len(seeds_of[m]) for m in members)
        cost = sum(infer_ms[m] for m in members)
        # equal-weight, seed-matched
        sm = [summarize(Y, np.mean([avail[m][k] for m in members], axis=0), labels, groups, n_boot=0)[1]
              for k in range(n_seed)]
        au = [s["macro_auroc_primary"] for s in sm]
        rows.append({"candidate": cname, "method": "mean", "seeds": "seed-matched mean+-SD",
                     "macro_auroc": np.mean(au), "auroc_sd": np.std(au, ddof=1) if len(au) > 1 else np.nan,
                     "macro_auprc": np.mean([s["macro_auprc_primary"] for s in sm]),
                     "n_seeds": n_seed, "members": "+".join(members), "cpu_ms_per_record": cost})
        seed_avg = [np.mean(avail[m], axis=0) for m in members]
        for method in ("mean", "weighted", "stacking"):
            if method == "mean":
                P = np.mean(seed_avg, axis=0)
                note = "no fitting; full fold 9"
            else:
                P = cross_validated(seed_avg, Y, groups, method)
                note = "5-fold patient-grouped CV within fold 9, pooled"
            _, s = summarize(Y, P, labels, groups)
            # full-fold-9 fit of the combiner (for later fold-10 use) and its predictions
            if method == "weighted":
                W = fit_weights(seed_avg, Y)
                P_full, combiner = apply_weights(seed_avg, W), {"weights": W.tolist()}
            elif method == "stacking":
                st = fit_stacker(seed_avg, Y)
                P_full = apply_stacker(seed_avg, st)
                combiner = {"stacking": [None if m is None else
                                         {"coef": m.coef_[0].tolist(), "intercept": float(m.intercept_[0])}
                                         for m in st]}
            else:
                P_full, combiner = P, {}
            ens_preds[(cname, method)] = (P, P_full, {"method": method, "members": members,
                                                      "member_seeds": {m: seeds_of[m] for m in members},
                                                      **combiner})
            rows.append({"candidate": cname, "method": method, "seeds": "seed-averaged",
                         "macro_auroc": s["macro_auroc_primary"], "ci_low": s["ci_low"],
                         "ci_high": s["ci_high"], "macro_auprc": s["macro_auprc_primary"],
                         "n_seeds": n_seed, "members": "+".join(members), "evaluation": note,
                         "cpu_ms_per_record": sum(infer_ms[m] * len(seeds_of[m]) for m in members)})

    res = pd.DataFrame(rows)
    res["gain_vs_best_single"] = res["macro_auroc"] - best_auroc
    is_ens = ~res["method"].isin(["single model", "seed average"])
    res["exceeds_sd_rule"] = is_ens & (res["gain_vs_best_single"] > best_sd)
    (RESULTS / "ensembles").mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS / "ensembles" / "ensemble_results.csv", index=False)
    pd.set_option("display.width", 250)
    print(f"best single model: {best_single} {best_auroc:.4f} +- {best_sd:.4f}")
    print(res.drop(columns=[c for c in ("evaluation",) if c in res]).round(4).to_string(index=False))

    # ---- calibration and thresholds ------------------------------------------------------------
    # (CV-estimated predictions, full-fold-9 predictions, combiner spec)
    targets = {"resnet1d_seed0": (avail["resnet1d"][0], avail["resnet1d"][0],
                                  {"method": "single", "members": ["resnet1d"], "member_seeds": {"resnet1d": [0]}})}
    # the two candidates pre-declared for the fold-10 test
    targets["resnet1d_3seed_avg"] = (np.mean(avail["resnet1d"], axis=0), np.mean(avail["resnet1d"], axis=0),
                                     {"method": "seed_average", "members": ["resnet1d"],
                                      "member_seeds": {"resnet1d": seeds_of["resnet1d"]}})
    if ("resnet1d+lightgbm", "mean") in ens_preds:
        targets["ensemble_resnet1d+lightgbm_mean"] = ens_preds[("resnet1d+lightgbm", "mean")]
    ens_rows = res[res["exceeds_sd_rule"] & ~res["seeds"].str.startswith("seed-matched")]
    if len(ens_rows):
        top = ens_rows.sort_values("macro_auroc", ascending=False).iloc[0]
        targets[f"ensemble_{top['candidate']}_{top['method']}"] = ens_preds[(top["candidate"], top["method"])]
        print(f"best retained-candidate ensemble: {top['candidate']} / {top['method']}")
    out_dir = RESULTS / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, (P, P_full, combiner) in targets.items():
        P_cal, chosen, _ = cal.cv_calibrate(P, Y, groups)
        ctab = pd.DataFrame(cal.calibration_table(Y, P, P_cal, ids))
        ctab.insert(1, "method", chosen)
        ctab.to_csv(out_dir / f"{name}_calibration.csv", index=False)
        th_out = {}
        for policy in ("max_f1", "sens90"):
            pred = cal.cv_thresholds(P_cal, Y, groups, ids, policy)
            rows_b, agg = cal.binary_metrics(Y.astype(int), pred.astype(int), ids, tiers)
            pd.DataFrame(rows_b).to_csv(out_dir / f"{name}_thresholds_{policy}_cv.csv", index=False)
            th_out[policy] = (pd.DataFrame(rows_b), agg)
        # fitted on full fold 9 for later fold-10 use: combiner, then calibrators on its outputs
        fitted = cal.fit_full(P_full, Y, ids, chosen)
        fitted["fitted_on"] = "fold 9 (all records)"
        fitted["predictor"] = name
        fitted["combiner"] = combiner
        with open(out_dir / f"{name}_fitted.json", "w", encoding="utf-8") as fh:
            json.dump(fitted, fh, indent=2)
        summary[name] = (ctab, th_out)
        print(f"\n=== {name}: calibration (CV) ===")
        print(ctab.round(4).to_string(index=False))
        for policy, (tab, agg) in th_out.items():
            print(f"--- thresholds {policy} (CV): {json.dumps({k: round(v, 4) for k, v in agg.items()})}")
            print(tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
