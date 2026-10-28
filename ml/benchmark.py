"""
PTB-XL benchmark. Trains on folds 1-8, evaluates on fold 9 (validation). Fold 10 is never read here.

    python benchmark.py                       # all classical runs, reuse features and finished runs
    python benchmark.py --models logreg,lightgbm --seeds 0,1,2
    python benchmark.py --models simple_cnn,resnet1d
    python benchmark.py --models simple_cnn --dev-subset --epochs 2 --smoke   # timing smoke test
    python benchmark.py --force               # rebuild features / retrain finished runs

Outputs (ml/results/): benchmark_results.csv and per_class_metrics.csv (appended),
predictions/<run_id>.npz (fold-9 scores + labels), models/<run_id>.joblib (classical) or
models/<run_id>/ (deep: best.pt, last.pt, normalizer.json, config.yaml, history.csv).
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
    "simple_cnn": ("simple_cnn", "raw100"),
    "resnet1d": ("resnet1d", "raw100"),
}
DEEP = {"simple_cnn", "resnet1d"}
DEFAULT_MODELS = ["logreg", "random_forest", "extra_trees", "lightgbm", "lightgbm_agesex"]
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
    if model in DEEP:
        params = {**config["deep_models"], "preprocessing": config["preprocessing"]}
        features = feature_set
    else:
        params, features = model_params(model), feature_names()
    payload = {"data": config["data"], "splits": config["splits"],
               "dev_subset": config.get("dev_subset"), "model": model,
               "params": params, "feature_set": feature_set, "features": features}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]


def append_csv(df, path):
    """Append rows; if the columns changed (e.g. a new column), rewrite with the union of columns."""
    if path.exists():
        old_cols = list(pd.read_csv(path, nrows=0).columns)
        if old_cols != list(df.columns):
            merged = pd.concat([pd.read_csv(path), df], ignore_index=True)
            if "smoke" in merged:
                merged["smoke"] = [bool(v) if pd.notna(v) else False for v in merged["smoke"]]
            merged.to_csv(path, index=False)
            return
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def finished_runs(path):
    if not path.exists():
        return set()
    return set(pd.read_csv(path)["run_id"])


def run_classical(model, feature_set, seed, run_id, ctx):
    X_train = feature_matrix(ctx["feats"], ctx["train"], feature_set)
    X_val = feature_matrix(ctx["feats"], ctx["val"], feature_set)
    clf = PerLabelClassifier(model, ctx["ids"], seed=seed)
    t0 = time.time()
    clf.fit(X_train.values, ctx["Y_train"], groups=ctx["train"]["patient_id"].values)
    train_time = time.time() - t0
    t0 = time.time()
    P_val = clf.predict_proba(X_val.values)
    infer_ms = (time.time() - t0) * 1000 / len(X_val)
    model_path = RESULTS_DIR / "models" / f"{run_id}.joblib"
    joblib.dump({"model": clf, "feature_columns": list(X_train.columns), "labels": ctx["ids"]},
                model_path)
    return {"scores": P_val, "Y_val": ctx["Y_val"], "val_meta": ctx["val"],
            "train_time_s": train_time, "inference_ms_per_record": infer_ms,
            "model_size_mb": model_path.stat().st_size / 2**20, "extra": {}}


def run_deep(model, seed, run_id, ctx, epochs, force):
    from src.train import train_deep
    res = train_deep(model, ctx["config"], ctx["dataset"], ctx["labels"], seed,
                     RESULTS_DIR / "models" / run_id, epochs=epochs, force=force,
                     log=lambda s: print(s, flush=True))
    res["model_size_mb"] = Path(res["model_path"]).stat().st_size / 2**20
    res["extra"] = {"n_params": res["n_params"], "best_epoch": res["best_epoch"],
                    "epochs_run": res["epochs_run"], "train_records": res["train_records"],
                    "mean_epoch_time_s": round(res["mean_epoch_time_s"], 1),
                    "mean_eval_time_s": round(res["mean_eval_time_s"], 1)}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=str(ML_DIR / "configs" / "benchmark.yaml"))
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS), help=f"comma list from {list(RUNS)}")
    ap.add_argument("--dev-subset", action="store_true",
                    help="use dev_subset.fraction of the splits in dev_subset.apply_to")
    ap.add_argument("--seeds", default=None,
                    help="comma list of seeds (default: first seed in config)")
    ap.add_argument("--epochs", type=int, default=None, help="deep models: override epochs")
    ap.add_argument("--smoke", action="store_true",
                    help="mark rows smoke=true (timing/smoke test, not a benchmark result)")
    ap.add_argument("--force", action="store_true", help="rebuild features and redo finished runs")
    args = ap.parse_args()

    config = load_config(args.config)
    if args.dev_subset:
        config = copy.deepcopy(config)
        config["dev_subset"]["enabled"] = True
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [config["seeds"][0]]
    labels = load_label_config(ML_DIR / config["data"]["labels_config"])
    ids = label_ids(labels)
    cli_names = [m.strip() for m in args.models.split(",") if m.strip()]

    (RESULTS_DIR / "predictions").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "models").mkdir(parents=True, exist_ok=True)
    results_csv = RESULTS_DIR / "benchmark_results.csv"
    per_class_csv = RESULTS_DIR / "per_class_metrics.csv"

    dataset = PTBXL(config, sampling_rate=config["data"]["sampling_rate"])
    train = dataset.get_split("train", load_signals=False).meta
    val = dataset.get_split("val", load_signals=False).meta
    ctx = {"config": config, "dataset": dataset, "labels": labels, "ids": ids, "train": train,
           "val": val, "Y_train": build_label_matrix(train["scp_codes"], labels),
           "Y_val": build_label_matrix(val["scp_codes"], labels)}
    print(f"train {len(train)} records, val {len(val)} records", flush=True)
    if any(n not in DEEP for n in cli_names):
        ctx["feats"], finfo = load_or_build_features(dataset, config["data"]["features_path"],
                                                     force=args.force,
                                                     n_jobs=config["features"]["n_jobs"])
        print("features:", json.dumps(finfo, indent=1), flush=True)

    done = finished_runs(results_csv)
    for cli_name, seed in [(m, s) for s in seeds for m in cli_names]:
        model, feature_set = RUNS[cli_name]
        run_id = (f"{model}__{feature_set}__seed{seed}" + ("__dev" if args.dev_subset else "")
                  + (f"__ep{args.epochs}" if args.epochs and model in DEEP else "")
                  + ("__smoke" if args.smoke else ""))
        if run_id in done and not args.force:
            print(f"skip {run_id} (already in results)", flush=True)
            continue
        print(f"=== {run_id}", flush=True)
        try:
            if model in DEEP:
                res = run_deep(model, seed, run_id, ctx, args.epochs, args.force)
            else:
                res = run_classical(model, feature_set, seed, run_id, ctx)
            np.savez_compressed(RESULTS_DIR / "predictions" / f"{run_id}.npz", scores=res["scores"],
                                labels=res["Y_val"], ecg_id=res["val_meta"].index.values,
                                label_ids=np.array(ids))
            per_label, summary = evaluate(res["Y_val"], res["scores"], labels,
                                          res["val_meta"]["patient_id"].values, n_boot=1000, seed=0)
            row = {
                "run_id": run_id, "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": model, "feature_set": feature_set, "seed": seed, "split": "val",
                "macro_auroc_primary": round(summary["macro_auroc_primary"], 4),
                "ci_low": round(summary["ci_low"], 4), "ci_high": round(summary["ci_high"], 4),
                "macro_auprc_primary": round(summary["macro_auprc_primary"], 4),
                "n_labels_in_macro": summary["n_labels_in_macro"],
                "excluded_from_macro": ";".join(summary["excluded_from_macro"]),
                "train_time_s": round(res["train_time_s"], 1),
                "inference_ms_per_record": round(res["inference_ms_per_record"], 4),
                "model_size_mb": round(res["model_size_mb"], 2),
                "peak_rss_mb": peak_rss_mb(),
                "config_hash": config_hash(config, model, feature_set),
                "smoke": bool(args.smoke),
            }
            pc = per_label.reset_index()
            pc.insert(0, "run_id", run_id)
            pc.insert(1, "split", "val")
            append_csv(pc, per_class_csv)
            append_csv(pd.DataFrame([row]), results_csv)
            print(json.dumps({**row, **res["extra"]}, indent=1), flush=True)
            print(per_label[["tier", "n_pos", "auroc", "auprc"]].round(4).to_string(), flush=True)
        except Exception:
            print(f"FAILED {run_id}:\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
