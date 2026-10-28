"""
Classical baseline benchmark on PTB-XL. Trains on folds 1-8, evaluates on fold 9 (validation).
Fold 10 is never read here.

    python benchmark.py                       # all runs, reuse features and finished runs
    python benchmark.py --models logreg,lightgbm --dev-subset
    python benchmark.py --force               # rebuild features and redo finished runs

Outputs (ml/results/): benchmark_results.csv and per_class_metrics.csv (appended),
predictions/<run_id>.npz (fold-9 scores + labels), models/<run_id>.joblib.
"""
import argparse
import copy
import ctypes
import hashlib
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ML_DIR))

from src.data_loader import PTBXL, load_config  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.feature_extraction import feature_names, load_or_build_features  # noqa: E402
from src.labels import build_label_matrix, label_ids, load_label_config  # noqa: E402
from src.models.classical import PerLabelClassifier, model_params  # noqa: E402

RESULTS_DIR = ML_DIR / "results"
RUNS = {  # cli name -> (model, feature set)
    "logreg": ("logreg", "signal"),
    "random_forest": ("random_forest", "signal"),
    "extra_trees": ("extra_trees", "signal"),
    "lightgbm": ("lightgbm", "signal"),
    "lightgbm_agesex": ("lightgbm", "signal+age_sex"),
}
PTBXL_AGE_OVER_89 = 300     # PTB-XL encodes ages above 89 as 300


def peak_rss_mb():
    """Peak working set of this process (Windows), else None."""
    if sys.platform != "win32":
        return None

    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
    if not kernel32.K32GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        return None
    return round(pmc.PeakWorkingSetSize / 2**20, 1)


def feature_matrix(feats, meta, feature_set):
    X = feats.loc[meta.index, feature_names()].copy()
    if feature_set == "signal+age_sex":
        age = meta["age"].where(meta["age"] != PTBXL_AGE_OVER_89, 90)
        X["age"] = age.values
        X["sex"] = meta["sex"].values
    elif feature_set != "signal":
        raise ValueError(f"unknown feature set {feature_set!r}")
    return X


def config_hash(config, model, feature_set):
    payload = {"data": config["data"], "splits": config["splits"],
               "dev_subset": config.get("dev_subset"), "model": model,
               "params": model_params(model), "feature_set": feature_set,
               "features": feature_names()}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]


def append_csv(df, path):
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def finished_runs(path):
    if not path.exists():
        return set()
    return set(pd.read_csv(path)["run_id"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=str(ML_DIR / "configs" / "benchmark.yaml"))
    ap.add_argument("--models", default=",".join(RUNS), help=f"comma list from {list(RUNS)}")
    ap.add_argument("--dev-subset", action="store_true", help="use dev_subset.fraction of each split")
    ap.add_argument("--seeds", default=None,
                    help="comma list of seeds (default: first seed in config)")
    ap.add_argument("--force", action="store_true", help="rebuild features and redo finished runs")
    args = ap.parse_args()

    config = load_config(args.config)
    if args.dev_subset:
        config = copy.deepcopy(config)
        config["dev_subset"]["enabled"] = True
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [config["seeds"][0]]
    labels = load_label_config(ML_DIR / config["data"]["labels_config"])
    ids = label_ids(labels)

    (RESULTS_DIR / "predictions").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "models").mkdir(parents=True, exist_ok=True)
    results_csv = RESULTS_DIR / "benchmark_results.csv"
    per_class_csv = RESULTS_DIR / "per_class_metrics.csv"

    dataset = PTBXL(config, sampling_rate=config["data"]["sampling_rate"])
    feats, finfo = load_or_build_features(dataset, config["data"]["features_path"], force=args.force,
                                          n_jobs=config["features"]["n_jobs"])
    print("features:", json.dumps(finfo, indent=1), flush=True)

    train = dataset.get_split("train", load_signals=False).meta
    val = dataset.get_split("val", load_signals=False).meta
    Y_train = build_label_matrix(train["scp_codes"], labels)
    Y_val = build_label_matrix(val["scp_codes"], labels)
    print(f"train {len(train)} records, val {len(val)} records", flush=True)

    done = finished_runs(results_csv)
    cli_names = [m.strip() for m in args.models.split(",") if m.strip()]
    for cli_name, seed in [(m, s) for s in seeds for m in cli_names]:
        model, feature_set = RUNS[cli_name]
        run_id = f"{model}__{feature_set}__seed{seed}" + ("__dev" if args.dev_subset else "")
        if run_id in done and not args.force:
            print(f"skip {run_id} (already in results)", flush=True)
            continue
        print(f"=== {run_id}", flush=True)
        try:
            X_train = feature_matrix(feats, train, feature_set)
            X_val = feature_matrix(feats, val, feature_set)
            clf = PerLabelClassifier(model, ids, seed=seed)
            t0 = time.time()
            clf.fit(X_train.values, Y_train, groups=train["patient_id"].values)
            train_time = time.time() - t0
            t0 = time.time()
            P_val = clf.predict_proba(X_val.values)
            infer_ms = (time.time() - t0) * 1000 / len(X_val)

            model_path = RESULTS_DIR / "models" / f"{run_id}.joblib"
            joblib.dump({"model": clf, "feature_columns": list(X_train.columns), "labels": ids},
                        model_path)
            np.savez_compressed(RESULTS_DIR / "predictions" / f"{run_id}.npz", scores=P_val,
                                labels=Y_val, ecg_id=val.index.values, label_ids=np.array(ids))

            per_label, summary = evaluate(Y_val, P_val, labels, val["patient_id"].values,
                                          n_boot=1000, seed=0)
            row = {
                "run_id": run_id, "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": model, "feature_set": feature_set, "seed": seed, "split": "val",
                "macro_auroc_primary": round(summary["macro_auroc_primary"], 4),
                "ci_low": round(summary["ci_low"], 4), "ci_high": round(summary["ci_high"], 4),
                "macro_auprc_primary": round(summary["macro_auprc_primary"], 4),
                "n_labels_in_macro": summary["n_labels_in_macro"],
                "excluded_from_macro": ";".join(summary["excluded_from_macro"]),
                "train_time_s": round(train_time, 1),
                "inference_ms_per_record": round(infer_ms, 4),
                "model_size_mb": round(model_path.stat().st_size / 2**20, 2),
                "peak_rss_mb": peak_rss_mb(),
                "config_hash": config_hash(config, model, feature_set),
            }
            pc = per_label.reset_index()
            pc.insert(0, "run_id", run_id)
            pc.insert(1, "split", "val")
            append_csv(pc, per_class_csv)
            append_csv(pd.DataFrame([row]), results_csv)
            print(json.dumps(row, indent=1), flush=True)
            print(per_label[["tier", "n_pos", "auroc", "auprc"]].round(4).to_string(), flush=True)
        except Exception:
            print(f"FAILED {run_id}:\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
