"""Write ml/results/final/preregistration.json for the fold-10 candidates (run BEFORE any fold-10 access)."""
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
REPO = ML_DIR.parent
sys.path.insert(0, str(ML_DIR))

from src.data_loader import load_config  # noqa: E402
from src.labels import load_label_config  # noqa: E402

FEATURE_MS_PER_RECORD = 165.3 * 16 * 1000 / 21799   # 500 Hz feature extraction, CPU time per record


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(p):
    return Path(p).resolve().relative_to(REPO).as_posix()


def main():
    config = load_config()
    labels = load_label_config(ML_DIR / config["data"]["labels_config"])
    models = ML_DIR / "results" / "models"
    calib = ML_DIR / "results" / "calibration"
    resnet = [{"seed": s, "dir": rel(models / f"resnet1d__raw100__seed{s}")} for s in (0, 1, 2)]
    lgbm = [{"seed": s, "file": rel(models / f"lightgbm__signal__seed{s}.joblib")} for s in (0, 1, 2)]
    resnet_files = [f"{r['dir']}/{f}" for r in resnet for f in ("best.pt", "normalizer.json", "config.yaml")]
    lgbm_files = [r["file"] for r in lgbm]
    candidates = {
        "primary": {"name": "resnet1d3+lightgbm3_mean",
                    "definition": "equal-weight average of (ResNet1d seeds 0-2 averaged sigmoid scores) and "
                                  "(LightGBM seeds 0-2 averaged probabilities, signal features)",
                    "model_files": resnet_files + lgbm_files,
                    "fitted_file": rel(calib / "ensemble_resnet1d+lightgbm_mean_fitted.json")},
        "fallback": {"name": "resnet1d3_avg",
                     "definition": "ResNet1d seeds 0-2 averaged sigmoid scores",
                     "model_files": resnet_files,
                     "fitted_file": rel(calib / "resnet1d_3seed_avg_fitted.json")},
    }
    hashed = sorted(set(resnet_files + lgbm_files + [c["fitted_file"] for c in candidates.values()]
                        + [rel(config["data"]["features_path"]) if Path(config["data"]["features_path"]).resolve()
                           .is_relative_to(REPO) else None]
                        + [rel(ML_DIR / f) for f in ("evaluate_test.py", "src/calibration.py", "src/evaluate.py",
                                                      "src/preprocessing.py", "src/train.py",
                                                      "src/models/deep.py", "src/models/classical.py",
                                                      "configs/benchmark.yaml", "configs/labels.yaml")]) - {None})
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "ml"], capture_output=True,
                           text=True).stdout.strip().splitlines()
    prereg = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "one-time evaluation on PTB-XL fold 10 (test); nothing is refitted after this point",
        "git_head": head,
        "uncommitted_ml_files_at_registration": dirty,
        "candidates": candidates,
        "components": {"resnet1d": resnet, "lightgbm": lgbm},
        "features_path": config["data"]["features_path"],
        "features_sha256": sha256(config["data"]["features_path"]),
        "feature_pipeline_cpu_ms_per_record": FEATURE_MS_PER_RECORD,
        "labels": {"all": [l["id"] for l in labels],
                   "primary": [l["id"] for l in labels if l["tier"] == "primary"],
                   "exploratory": [l["id"] for l in labels if l["tier"] == "exploratory"],
                   "rule": "key presence in scp_codes, likelihood ignored"},
        "threshold_policies": {"max_f1": "per-label threshold maximising F1 on fold 9",
                               "sens90": "sensitivity >= 0.90 on fold 9 for AFIB, AFLT, SVT, AVB_HIGH, LBBB; "
                                         "max_f1 for the other labels"},
        "calibration": "per label, method (none/temperature/platt) chosen by 5-fold patient-grouped CV "
                       "log-loss on fold 9, then fitted on all of fold 9",
        "inference": "100 Hz, per-lead z-score (train-fold statistics), 2.5 s windows, 50% overlap, "
                     "per-label max over windows, sigmoid; LightGBM from the 500 Hz feature table",
        "metrics": ["macro-AUROC over the 9 primary labels with 1000x patient-level bootstrap 95% CI",
                    "macro-AUPRC (primary)", "per-label AUROC, AUPRC, prevalence",
                    "per policy: per-label sensitivity, specificity, precision, F1, 2x2 confusion matrix; "
                    "macro-F1 and micro-F1 over primary labels",
                    "ECE (15 equal-width bins) and Brier, raw vs calibrated",
                    "exploratory labels reported separately with n_pos and a warning",
                    "CPU inference ms/record and model size"],
        "sha256": {p: sha256(REPO / p) for p in hashed},
    }
    out = ML_DIR / "results" / "final" / "preregistration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prereg, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(prereg['sha256'])} hashed files, HEAD {head})")


if __name__ == "__main__":
    main()
