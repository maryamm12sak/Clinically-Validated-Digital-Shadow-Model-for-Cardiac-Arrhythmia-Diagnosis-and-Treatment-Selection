"""
Build the deployable model package in ml/deploy/ from the pre-registered final models.

    python export_deploy.py

ONNX export of the 3 ResNet1d seeds (input [batch, 12, 250] float32, output logits [batch, 12],
opset 17), LightGBM boosters in native text format, calibrators/thresholds for both modes,
manifest.json with SHA-256 hashes and fold-10 metrics. Parity checks use fold 9 only.
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
import yaml
from torch import nn

ML_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ML_DIR))

from benchmark import feature_matrix  # noqa: E402
from src.data_loader import PTBXL, load_config  # noqa: E402
from src.evaluate import evaluate  # noqa: E402
from src.labels import build_label_matrix, label_ids, load_label_config  # noqa: E402
from src.models.classical import ConstantClassifier  # noqa: E402
from src.models.deep import build_model  # noqa: E402
from src.preprocessing import PerLeadNormalizer, sliding_windows  # noqa: E402
from src.train import prepare_signals  # noqa: E402

DEPLOY = ML_DIR / "deploy"
MODELS = ML_DIR / "results" / "models"
VERSION = "ptbxl-arrhythmia-v1.0"
SEEDS = (0, 1, 2)


class ChannelsFirst(nn.Module):
    """ONNX wrapper: input (B, 12, T) instead of the training layout (B, T, 12)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x.transpose(1, 2))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_resnet(config, X_val, Y_val, groups, labels):
    out, report = [], {}
    window = 250
    seed_probs_torch, seed_probs_onnx = [], []
    for s in SEEDS:
        d = MODELS / f"resnet1d__raw100__seed{s}"
        cfg = yaml.safe_load(open(d / "config.yaml", encoding="utf-8"))
        model = build_model("resnet1d", n_classes=12, **cfg["model_kwargs"])
        model.load_state_dict(torch.load(d / "best.pt", map_location="cpu", weights_only=False)["model"])
        model.eval()
        wrapper = ChannelsFirst(model).eval()
        onnx_path = DEPLOY / "models" / f"resnet1d_seed{s}.onnx"
        torch.onnx.export(wrapper, (torch.randn(2, 12, window),), str(onnx_path),
                          input_names=["input"], output_names=["logits"],
                          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                          opset_version=17, dynamo=False)
        shutil.copy(d / "normalizer.json", DEPLOY / "models" / f"normalizer_seed{s}.json")

        norm = PerLeadNormalizer.load(d / "normalizer.json")
        Xn = prepare_signals(X_val, 100, config, norm)
        wins, owner = [], []
        for i, x in enumerate(Xn):
            w, _ = sliding_windows(x, window, 0.5)
            wins.append(w)
            owner += [i] * len(w)
        wins = np.concatenate(wins)
        owner = np.asarray(owner)
        with torch.no_grad():
            lt = torch.cat([model(torch.from_numpy(wins[i:i + 1024]))
                            for i in range(0, len(wins), 1024)]).numpy()
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        lo = np.concatenate([sess.run(None, {"input": np.ascontiguousarray(wins[i:i + 1024].transpose(0, 2, 1))})[0]
                             for i in range(0, len(wins), 1024)])
        rec_t = np.full((len(Xn), 12), -np.inf, np.float32)
        rec_o = np.full((len(Xn), 12), -np.inf, np.float32)
        np.maximum.at(rec_t, owner, lt)
        np.maximum.at(rec_o, owner, lo)
        seed_probs_torch.append(1 / (1 + np.exp(-rec_t.astype(np.float64))))
        seed_probs_onnx.append(1 / (1 + np.exp(-rec_o.astype(np.float64))))
        report[f"seed{s}"] = {"n_windows": int(len(wins)),
                              "max_abs_logit_diff": float(np.abs(lt - lo).max())}
        out.append({"seed": s, "onnx": f"models/resnet1d_seed{s}.onnx",
                    "normalizer": f"models/normalizer_seed{s}.json",
                    "source_checkpoint": f"ml/results/models/resnet1d__raw100__seed{s}/best.pt"})
    pt, po = np.mean(seed_probs_torch, axis=0), np.mean(seed_probs_onnx, axis=0)
    report["macro_auroc_fold9_torch"] = evaluate(Y_val, pt, labels, groups, n_boot=0)[1]["macro_auroc_primary"]
    report["macro_auroc_fold9_onnx"] = evaluate(Y_val, po, labels, groups, n_boot=0)[1]["macro_auroc_primary"]
    saved = np.mean([np.load(ML_DIR / "results" / "predictions" / f"resnet1d__raw100__seed{s}.npz")["scores"]
                     for s in SEEDS], axis=0)
    report["max_abs_prob_diff_vs_saved_fold9"] = float(np.abs(po - saved).max())
    return out, report, po


def export_lightgbm(config, ds, val_meta, labels):
    import lightgbm as lgb
    ids = label_ids(labels)
    feats = pd.read_csv(config["data"]["features_path"], index_col="ecg_id")
    X = feature_matrix(feats, val_meta, "signal")
    out, report, probs = [], {}, []
    cols = None
    for s in SEEDS:
        bundle = joblib.load(MODELS / f"lightgbm__signal__seed{s}.joblib")
        cols = bundle["feature_columns"]
        clf = bundle["model"]
        comp = {"seed": s, "labels": {}, "source": f"ml/results/models/lightgbm__signal__seed{s}.joblib"}
        p_ref = clf.predict_proba(X.values)
        p_txt = np.zeros_like(p_ref)
        for j, (lid, est) in enumerate(zip(ids, clf.estimators)):
            if isinstance(est, ConstantClassifier):
                comp["labels"][lid] = {"constant": est.p}
                p_txt[:, j] = est.p
                continue
            path = DEPLOY / "models" / f"lightgbm_seed{s}_{lid}.txt"
            best = est.best_iteration_ if getattr(est, "best_iteration_", None) else -1
            est.booster_.save_model(str(path), num_iteration=best)
            comp["labels"][lid] = {"file": f"models/lightgbm_seed{s}_{lid}.txt", "num_iteration": int(best)}
            p_txt[:, j] = lgb.Booster(model_file=str(path)).predict(X.values.astype(np.float32))
        report[f"seed{s}_max_abs_prob_diff_text_vs_sklearn"] = float(np.abs(p_ref - p_txt).max())
        probs.append(p_txt)
        out.append(comp)
    (DEPLOY / "feature_columns.json").write_text(json.dumps(cols, indent=1), encoding="utf-8")
    return out, report, np.mean(probs, axis=0)


def test_metrics():
    m = pd.read_csv(ML_DIR / "results" / "final" / "test_metrics.csv")
    pc = pd.read_csv(ML_DIR / "results" / "final" / "test_per_class.csv")
    out = {}
    for mode, cand in (("ensemble", "resnet1d3+lightgbm3_mean"), ("resnet_only", "resnet1d3_avg")):
        r = m[m["candidate"] == cand].iloc[0]
        p = pc[pc["candidate"] == cand]
        out[mode] = {
            "dataset": "PTB-XL v1.0.3 fold 10 (held-out test fold, evaluated once)",
            "n_records": int(r["n_records"]),
            "macro_auroc_primary": round(float(r["macro_auroc_primary"]), 4),
            "macro_auroc_95ci": [round(float(r["ci_low"]), 4), round(float(r["ci_high"]), 4)],
            "macro_auprc_primary": round(float(r["macro_auprc_primary"]), 4),
            "macro_f1_primary_max_f1": round(float(r["max_f1_macro_f1_primary"]), 4),
            "micro_f1_primary_max_f1": round(float(r["max_f1_micro_f1_primary"]), 4),
            "per_label": {row["label"]: {
                "tier": row["tier"], "n_pos": int(row["n_pos"]),
                "auroc": round(float(row["auroc"]), 4), "auprc": round(float(row["auprc"]), 4),
                "sensitivity_max_f1": round(float(row["max_f1_sensitivity"]), 4),
                "specificity_max_f1": round(float(row["max_f1_specificity"]), 4),
                "sensitivity_high_sens": round(float(row["sens90_sensitivity"]), 4),
                "specificity_high_sens": round(float(row["sens90_specificity"]), 4)}
                for _, row in p.iterrows()},
        }
    return out


def main():
    config = load_config()
    labels = load_label_config(ML_DIR / config["data"]["labels_config"])
    (DEPLOY / "models").mkdir(parents=True, exist_ok=True)
    ds = PTBXL(config, sampling_rate=100)
    val = ds.get_split("val")
    Y_val = build_label_matrix(val.meta["scp_codes"], labels)
    groups = val.meta["patient_id"].values

    resnet, r_report, p_res = export_resnet(config, val.X, Y_val, groups, labels)
    lgbm, l_report, p_lgb = export_lightgbm(config, ds, val.meta, labels)
    saved_ens = np.mean([np.mean([np.load(ML_DIR / "results" / "predictions" / f"{m}__seed{s}.npz")["scores"]
                                  for s in SEEDS], axis=0)
                         for m in ("resnet1d__raw100", "lightgbm__signal")], axis=0)
    ens_diff = float(np.abs((p_res + p_lgb) / 2 - saved_ens).max())

    shutil.copy(ML_DIR / "configs" / "labels.yaml", DEPLOY / "labels.yaml")
    cal_dir = ML_DIR / "results" / "calibration"
    shutil.copy(cal_dir / "ensemble_resnet1d+lightgbm_mean_fitted.json", DEPLOY / "calibration_ensemble.json")
    shutil.copy(cal_dir / "resnet1d_3seed_avg_fitted.json", DEPLOY / "calibration_resnet_only.json")

    head = subprocess.run(["git", "-C", str(ML_DIR.parent), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    files = sorted(p for p in DEPLOY.rglob("*") if p.is_file() and p.suffix in
                   (".onnx", ".txt", ".json", ".yaml", ".py") and p.name != "manifest.json"
                   and "__pycache__" not in p.parts)
    manifest = {
        "model_version": VERSION,
        "default_mode": "resnet_only",
        "modes": {"resnet_only": "shipped MVP model: ResNet1d, 3 seeds averaged (ONNX)",
                  "ensemble": "documented upgrade: mean of ResNet1d (3 seeds) and LightGBM (3 seeds); "
                              "needs lightgbm, neurokit2, PyWavelets and the 500 Hz feature pipeline"},
        "training_commit": head,
        "preregistration_commit": "14486d8",
        "labels": label_ids(labels),
        "input": {"shape": "(samples, 12)", "units": "mV", "lead_order": ["I", "II", "III", "aVR", "aVL",
                  "aVF", "V1", "V2", "V3", "V4", "V5", "V6"], "fs": "any (resampled internally)",
                  "duration_s": [5, 60]},
        "onnx": {"input": "input float32 [batch, 12, 250] (one 2.5 s window at 100 Hz, per-lead z-scored)",
                 "output": "logits float32 [batch, 12]", "opset": 17},
        "components": {"resnet1d": resnet, "lightgbm": lgbm},
        "parity_fold9": {"resnet": r_report, "lightgbm": l_report,
                         "ensemble_max_abs_prob_diff_vs_saved": ens_diff},
        "test_metrics": test_metrics(),
        "sha256": {p.relative_to(DEPLOY).as_posix(): sha256(p) for p in files},
    }
    (DEPLOY / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total = sum(p.stat().st_size for p in files) / 2**20
    print(json.dumps(manifest["parity_fold9"], indent=1))
    print(f"{len(files)} files, {total:.1f} MB")


if __name__ == "__main__":
    main()
