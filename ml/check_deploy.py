"""
End-to-end parity of the deploy package (ml/deploy) on fold 9, through the deploy code only.

    python check_deploy.py

1. Component parity: ResNet path fed PTB-XL's 100 Hz files and LightGBM path fed the 500 Hz files
   must reproduce the saved fold-9 predictions.
2. End-to-end: every fold-9 record from its raw 500 Hz WFDB file through predict_detailed (both modes);
   reports the difference to the saved predictions (the ResNet path then resamples 500 -> 100 Hz
   itself instead of using PTB-XL's own 100 Hz files) and the macro-AUROC.
3. The backend demo files (samples/*.csv, ecg_00001_500hz.csv): top predictions and runtime.
Fold 10 is not read here (the demo samples are fold-10 records chosen by make_samples.py; running
them only produces predictions, nothing is fitted).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from joblib import Parallel, delayed

ML_DIR = Path(__file__).resolve().parent
REPO = ML_DIR.parent
sys.path.insert(0, str(ML_DIR))
sys.path.insert(0, str(ML_DIR / "deploy"))

import ecg_model  # noqa: E402
from src.data_loader import PTBXL, load_config  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.labels import build_label_matrix, load_label_config  # noqa: E402

PRED = ML_DIR / "results" / "predictions"


def saved(prefix):
    return np.mean([np.load(PRED / f"{prefix}__seed{s}.npz")["scores"] for s in (0, 1, 2)], axis=0)


def run_record(root, rel_hr, rel_lr):
    ens = ecg_model.ECGModel("ensemble")
    sig500, _ = wfdb.rdsamp(str(root / rel_hr))
    sig100, _ = wfdb.rdsamp(str(root / rel_lr))
    d = ens.predict_detailed(sig500, 500)
    ids = ens.label_ids
    comp_res = ens.resnet_probability(sig100, 100)
    comp_lgb, _ = ens.lightgbm_probability(sig500, 500)
    return {"e2e_resnet": [d["labels"][l]["resnet_probability"] for l in ids],
            "e2e_lgbm": [d["labels"][l]["lightgbm_probability"] for l in ids],
            "comp_resnet": comp_res.tolist(), "comp_lgbm": comp_lgb.tolist(), "ms": d["runtime_ms"]}


def main():
    config = load_config()
    labels = load_label_config(ML_DIR / config["data"]["labels_config"])
    ds = PTBXL(config, sampling_rate=100)
    val = ds.get_split("val", load_signals=False).meta
    Y = build_label_matrix(val["scp_codes"], labels)
    groups = val["patient_id"].values
    root = Path(config["data"]["data_root"])

    t0 = time.time()
    rows = Parallel(n_jobs=8)(delayed(run_record)(root, val.at[e, "filename_hr"], val.at[e, "filename_lr"])
                              for e in val.index)
    wall = time.time() - t0
    get = lambda k: np.array([r[k] for r in rows])
    s_res, s_lgb = saved("resnet1d__raw100"), saved("lightgbm__signal")
    s_ens = (s_res + s_lgb) / 2
    out = {
        "component_parity": {
            "resnet_max_abs_diff": float(np.abs(get("comp_resnet") - s_res).max()),
            "lightgbm_max_abs_diff": float(np.abs(get("comp_lgbm") - s_lgb).max()),
            "ensemble_max_abs_diff": float(np.abs((get("comp_resnet") + get("comp_lgbm")) / 2 - s_ens).max()),
        },
        "end_to_end_from_500hz": {},
        "fold9_records": len(val), "wall_s": round(wall, 1),
        "mean_ms_per_record_ensemble_single_process": round(float(np.mean(get("ms"))), 1),
    }
    for name, P, ref in (("resnet_only", get("e2e_resnet"), s_res),
                         ("ensemble", (get("e2e_resnet") + get("e2e_lgbm")) / 2, s_ens)):
        diff = np.abs(P - ref)
        out["end_to_end_from_500hz"][name] = {
            "max_abs_prob_diff": float(diff.max()), "mean_abs_prob_diff": float(diff.mean()),
            "p99_abs_prob_diff": float(np.percentile(diff, 99)),
            "macro_auroc_fold9": evaluate(Y, P, labels, groups, n_boot=0)[1]["macro_auroc_primary"],
            "macro_auroc_fold9_saved": evaluate(Y, ref, labels, groups, n_boot=0)[1]["macro_auroc_primary"],
        }
    out["lightgbm_e2e_max_abs_diff"] = float(np.abs(get("e2e_lgbm") - s_lgb).max())

    # demo files
    index = json.loads((REPO / "samples" / "index.json").read_text(encoding="utf-8"))
    demos = [(s["file"], s["fs"], s["description"]) for s in index] + [("../ecg_00001_500hz.csv", 500, "PTB-XL ecg 1 (500 Hz)")]
    res = ecg_model.ECGModel("resnet_only")
    out["demo"] = []
    for fname, fs, desc in demos:
        df = pd.read_csv(REPO / "samples" / fname)
        sig = df.values.astype(float)
        if sig.shape[1] != 12:
            sig = sig[:, -12:]
        try:
            d = res.predict_detailed(sig, fs)
            top = sorted(d["labels"].items(), key=lambda kv: -kv[1]["calibrated_probability"])[:3]
            out["demo"].append({"file": fname, "expected": desc, "fs": fs, "runtime_ms": round(d["runtime_ms"], 1),
                                "top3": [(k, round(v["calibrated_probability"], 3), bool(v["detected"])) for k, v in top]})
        except Exception as e:
            out["demo"].append({"file": fname, "error": repr(e)})
    (ML_DIR / "results" / "deploy").mkdir(parents=True, exist_ok=True)
    (ML_DIR / "results" / "deploy" / "deploy_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
