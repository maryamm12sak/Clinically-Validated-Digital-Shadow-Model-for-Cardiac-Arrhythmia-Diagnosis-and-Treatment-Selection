"""
One-time fold-10 evaluation of the pre-registered candidates (ml/results/final/preregistration.json).

    python evaluate_test.py --dry-run-val    # full pipeline on fold 9 only (checks the code; no fold-10 access)
    python evaluate_test.py                  # verifies hashes, then reads fold 10 once

Nothing is fitted here: models, calibrators and thresholds are loaded as pre-registered.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

ML_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ML_DIR))

from benchmark import feature_matrix  # noqa: E402
from src import calibration as cal  # noqa: E402
from src.data_loader import PTBXL, load_config  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.labels import build_label_matrix, label_ids, load_label_config  # noqa: E402
from src.models.deep import build_model  # noqa: E402
from src.preprocessing import PerLeadNormalizer  # noqa: E402
from src.train import predict_records, prepare_signals  # noqa: E402

FINAL = ML_DIR / "results" / "final"
PLOTS = ML_DIR / "results" / "plots"
PREREG = FINAL / "preregistration.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_hashes(prereg):
    bad = [p for p, h in prereg["sha256"].items() if sha256(ML_DIR.parent / p) != h]
    if sha256(prereg["features_path"]) != prereg["features_sha256"]:
        bad.append(prereg["features_path"])
    if bad:
        raise SystemExit(f"hash mismatch, refusing to evaluate: {bad}")
    print(f"verified {len(prereg['sha256'])} file hashes", flush=True)


def resnet_scores(prereg, X, config, fs):
    """Seed-averaged sigmoid scores of the pre-registered ResNet1d checkpoints."""
    window = int(round(config["preprocessing"]["window_s"] * config["data"]["sampling_rate"]))
    overlap = config["preprocessing"]["inference_overlap"]
    torch.set_num_threads(int(config["deep_models"]["num_threads"]))
    scores = []
    for run in prereg["components"]["resnet1d"]:
        d = ML_DIR.parent / run["dir"]
        run_cfg = yaml.safe_load(open(d / "config.yaml", encoding="utf-8"))
        model = build_model("resnet1d", n_classes=12, **run_cfg["model_kwargs"])
        model.load_state_dict(torch.load(d / "best.pt", map_location="cpu", weights_only=False)["model"])
        norm = PerLeadNormalizer.load(d / "normalizer.json")
        Xp = prepare_signals(X, fs, config, norm)
        logits = predict_records(model, Xp, window, overlap, config["deep_models"]["eval_batch_size"])
        scores.append(1 / (1 + np.exp(-logits)))
    return np.mean(scores, axis=0)


def lightgbm_scores(prereg, meta, config):
    feats = pd.read_csv(config["data"]["features_path"], index_col="ecg_id")
    X = feature_matrix(feats, meta, "signal")
    scores = []
    for run in prereg["components"]["lightgbm"]:
        bundle = joblib.load(ML_DIR.parent / run["file"])
        assert bundle["feature_columns"] == list(X.columns)
        scores.append(bundle["model"].predict_proba(X.values))
    return np.mean(scores, axis=0)


def apply_fitted(P, fitted):
    return np.stack([cal.apply_calibrator(P[:, j], c) for j, c in enumerate(fitted["calibrators"])], axis=1)


def candidate_metrics(name, Y, P_raw, fitted, labels, groups, split):
    ids = label_ids(labels)
    tiers = [lab["tier"] for lab in labels]
    P_cal = apply_fitted(P_raw, fitted)
    per_label, s = evaluate(Y, P_cal, labels, groups, n_boot=1000, seed=0)
    row = {"candidate": name, "split": split, "n_records": len(Y),
           "macro_auroc_primary": s["macro_auroc_primary"], "ci_low": s["ci_low"], "ci_high": s["ci_high"],
           "macro_auprc_primary": s["macro_auprc_primary"], "n_labels_in_macro": s["n_labels_in_macro"],
           "excluded_from_macro": ";".join(s["excluded_from_macro"])}
    pc = per_label.reset_index()[["label", "tier", "n", "n_pos", "prevalence", "auroc", "auprc", "undefined"]]
    for policy in ("max_f1", "sens90"):
        th = np.array(fitted["thresholds"][policy])
        pred = (P_cal >= th).astype(int)
        rows, agg = cal.binary_metrics(Y.astype(int), pred, ids, tiers)
        b = pd.DataFrame(rows)
        b["tn"] = len(Y) - b["tp"] - b["fp"] - b["fn"]
        b["threshold"] = th
        for col in ("threshold", "sensitivity", "specificity", "precision", "f1", "tp", "fp", "fn", "tn"):
            pc[f"{policy}_{col}"] = b[col].values
        row[f"{policy}_macro_f1_primary"] = agg["macro_f1_primary"]
        row[f"{policy}_micro_f1_primary"] = agg["micro_f1_primary"]
    ct = pd.DataFrame(cal.calibration_table(Y, P_raw, P_cal, ids))
    for col in ("ece_before", "ece_after", "brier_before", "brier_after"):
        pc[col.replace("before", "raw").replace("after", "calibrated")] = ct[col].values
    prim = pc["tier"] == "primary"
    row["mean_ece_raw_primary"] = pc.loc[prim, "ece_raw"].mean()
    row["mean_ece_calibrated_primary"] = pc.loc[prim, "ece_calibrated"].mean()
    row["mean_brier_raw_primary"] = pc.loc[prim, "brier_raw"].mean()
    row["mean_brier_calibrated_primary"] = pc.loc[prim, "brier_calibrated"].mean()
    pc["warning"] = np.where(pc["tier"] == "exploratory",
                             "exploratory: too few positives for reliable estimates", "")
    pc.insert(0, "candidate", name)
    pc.insert(1, "split", split)
    return row, pc, P_cal


def plots(name, Y, P_raw, P_cal, labels, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    PLOTS.mkdir(parents=True, exist_ok=True)
    ids = label_ids(labels)
    for kind in ("roc", "pr"):
        fig, axes = plt.subplots(3, 4, figsize=(16, 11))
        for j, ax in enumerate(axes.flat):
            y, p = Y[:, j], P_cal[:, j]
            if 0 < y.sum() < len(y):
                if kind == "roc":
                    fpr, tpr, _ = roc_curve(y, p)
                    ax.plot(fpr, tpr)
                    ax.plot([0, 1], [0, 1], "--", color="grey", lw=0.8)
                    ax.set_xlabel("1 - specificity")
                    ax.set_ylabel("sensitivity")
                else:
                    prec, rec, _ = precision_recall_curve(y, p)
                    ax.plot(rec, prec)
                    ax.axhline(y.mean(), ls="--", color="grey", lw=0.8)
                    ax.set_xlabel("recall")
                    ax.set_ylabel("precision")
            tier = labels[j]["tier"]
            ax.set_title(f"{ids[j]} (n_pos={int(y.sum())}{', exploratory' if tier != 'primary' else ''})")
        fig.suptitle(f"{name} - {kind.upper()} curves ({tag})")
        fig.tight_layout()
        fig.savefig(PLOTS / f"{name}_{kind}_{tag}.png", dpi=110)
        plt.close(fig)
    fig, axes = plt.subplots(3, 4, figsize=(16, 11))
    bins = np.linspace(0, 1, 16)
    for j, ax in enumerate(axes.flat):
        y = Y[:, j]
        for P, lab in ((P_raw, "raw"), (P_cal, "calibrated")):
            p = P[:, j]
            idx = np.minimum(np.digitize(p, bins) - 1, 14)
            xs, ys = [], []
            for b in range(15):
                m = idx == b
                if m.sum() >= 5:
                    xs.append(p[m].mean())
                    ys.append(y[m].mean())
            ax.plot(xs, ys, "o-", ms=3, label=lab)
        ax.plot([0, 1], [0, 1], "--", color="grey", lw=0.8)
        ax.set_title(f"{ids[j]} (n_pos={int(y.sum())})")
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed frequency")
        ax.legend(fontsize=7)
    fig.suptitle(f"{name} - reliability before/after calibration ({tag}, bins with >=5 records)")
    fig.tight_layout()
    fig.savefig(PLOTS / f"{name}_reliability_{tag}.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run-val", action="store_true", help="run the whole pipeline on fold 9 instead")
    args = ap.parse_args()

    prereg = json.load(open(PREREG, encoding="utf-8"))
    verify_hashes(prereg)
    config = load_config()
    labels = load_label_config(ML_DIR / config["data"]["labels_config"])
    assert label_ids(labels) == prereg["labels"]["all"]
    dataset = PTBXL(config, sampling_rate=100)

    if args.dry_run_val:
        split, tag = dataset.get_split("val"), "fold9_dryrun"
    else:
        done = FINAL / "test_access_log.txt"
        if done.exists() and "evaluation complete" in done.read_text(encoding="utf-8"):
            raise SystemExit("fold 10 has already been evaluated; refusing to run again")
        with open(done, "a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now(timezone.utc).isoformat()} fold 10 accessed via "
                     f"get_split('test', allow_test=True)\n")
        split, tag = dataset.get_split("test", allow_test=True), "fold10"
    meta, X = split.meta, split.X
    Y = build_label_matrix(meta["scp_codes"], labels)
    groups = meta["patient_id"].values
    print(f"{tag}: {len(meta)} records, folds {sorted(meta['strat_fold'].unique())}", flush=True)

    t0 = time.time()
    P_res = resnet_scores(prereg, X, config, dataset.sampling_rate)
    t_res = (time.time() - t0) * 1000 / len(meta)
    t0 = time.time()
    P_lgb = lightgbm_scores(prereg, meta, config)
    t_lgb = (time.time() - t0) * 1000 / len(meta)
    raw = {"primary": np.mean([P_res, P_lgb], axis=0), "fallback": P_res}
    cost = {"primary": t_res + t_lgb + prereg["feature_pipeline_cpu_ms_per_record"], "fallback": t_res}
    size = {k: sum((ML_DIR.parent / f).stat().st_size for f in prereg["candidates"][k]["model_files"]) / 2**20
            for k in raw}

    rows, per_class = [], []
    for key, P in raw.items():
        fitted = json.load(open(ML_DIR.parent / prereg["candidates"][key]["fitted_file"], encoding="utf-8"))
        row, pc, P_cal = candidate_metrics(prereg["candidates"][key]["name"], Y, P, fitted, labels, groups, tag)
        row["cpu_ms_per_record"] = cost[key]
        row["model_size_mb"] = size[key]
        rows.append(row)
        per_class.append(pc)
        plots(key, Y, P, P_cal, labels, tag)
        np.savez_compressed(FINAL / f"{key}_{tag}_predictions.npz", scores_raw=P, scores_calibrated=P_cal,
                            labels=Y, ecg_id=meta.index.values, label_ids=np.array(label_ids(labels)))
    metrics = pd.DataFrame(rows)
    per_class = pd.concat(per_class, ignore_index=True)
    suffix = "" if tag == "fold10" else f"_{tag}"
    metrics.to_csv(FINAL / f"test_metrics{suffix}.csv", index=False)
    per_class.to_csv(FINAL / f"test_per_class{suffix}.csv", index=False)
    pd.set_option("display.width", 250)
    print(metrics.round(4).T.to_string())
    print(per_class.round(3).to_string(index=False))
    if tag == "fold10":
        with open(FINAL / "test_access_log.txt", "a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now(timezone.utc).isoformat()} evaluation complete\n")


if __name__ == "__main__":
    main()
