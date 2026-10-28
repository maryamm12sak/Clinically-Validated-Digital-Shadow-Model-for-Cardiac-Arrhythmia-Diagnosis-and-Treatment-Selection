"""
Deep-model training on the 100 Hz cache (settings in benchmark.yaml -> deep_models).

- Every cached record goes through preprocessing.preprocess() (the same function used at inference).
- PerLeadNormalizer is fitted on the training folds only and saved next to the checkpoint.
- Training: random window_s crops, BCEWithLogitsLoss, AdamW + OneCycleLR.
- Validation (fold 9) every epoch: sliding window_s windows with inference_overlap, per-label
  max over a record's windows (Strodthoff et al. 2021); early stopping on macro-AUROC over the
  primary labels.
- Run directory ml/results/models/<run_id>/: best.pt, last.pt (resume), normalizer.json,
  config.yaml, history.csv.
"""
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from .evaluate import evaluate, macro_summary, per_label_metrics
from .labels import build_label_matrix, label_ids
from .models.deep import build_model, count_params
from .preprocessing import PerLeadNormalizer, preprocess, random_crop, sliding_windows


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_signals(X, fs, config, normalizer=None):
    """Run each record through preprocess() (cache is already at target fs), then normalize.
    Works on a copy: the caller's array is never modified."""
    target_fs = config["data"]["sampling_rate"]
    bp = config["preprocessing"]["bandpass"]
    X = np.array(X, dtype=np.float32, copy=True)
    for i in range(len(X)):
        X[i] = preprocess(X[i], fs, target_fs, bandpass_cfg=bp)
    if normalizer is not None:
        for i in range(0, len(X), 1024):
            X[i:i + 1024] = normalizer.transform(X[i:i + 1024])
    return X


@torch.no_grad()
def predict_records(model, X, window, overlap=0.5, batch_size=512, device="cpu"):
    """Max-aggregated logits per record (N, n_labels) from sliding windows over each record."""
    model.eval()
    windows, owner = [], []
    for i, x in enumerate(X):
        w, _ = sliding_windows(x, window, overlap)
        windows.append(w)
        owner.extend([i] * len(w))
    windows = np.concatenate(windows)
    owner = np.asarray(owner)
    logits = []
    for s in range(0, len(windows), batch_size):
        logits.append(model(torch.from_numpy(windows[s:s + batch_size]).to(device)).float().cpu())
    logits = torch.cat(logits).numpy()
    out = np.full((len(X), logits.shape[1]), -np.inf, dtype=np.float32)
    np.maximum.at(out, owner, logits)
    return out


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def train_deep(model_name, config, dataset, labels, seed, run_dir, epochs=None, force=False,
               log=print):
    """
    Train one deep model; returns dict with the fitted model, fold-9 scores/labels/meta and timings.
    """
    dcfg = config["deep_models"]
    epochs = epochs or dcfg["epochs"]
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(dcfg.get("num_threads") or max(1, (os.cpu_count() or 2) // 2)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(seed)

    fs = dataset.sampling_rate
    window = int(round(config["preprocessing"]["window_s"] * config["data"]["sampling_rate"]))
    overlap = config["preprocessing"]["inference_overlap"]

    train = dataset.get_split("train")
    val = dataset.get_split("val")
    Y_train = build_label_matrix(train.meta["scp_codes"], labels).astype(np.float32)
    Y_val = build_label_matrix(val.meta["scp_codes"], labels)

    X_train = prepare_signals(train.X, fs, config)
    normalizer = PerLeadNormalizer().fit(X_train)          # training folds only
    normalizer.save(run_dir / "normalizer.json")
    X_train = normalizer.transform(X_train)
    X_val = prepare_signals(val.X, fs, config, normalizer)
    train.X = val.X = None

    model_kwargs = dict(dcfg.get(model_name, {}))
    model = build_model(model_name, n_classes=len(labels), **model_kwargs).to(device)
    n_params = count_params(model)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump({"model": model_name, "model_kwargs": model_kwargs, "seed": seed,
                        "epochs": epochs, "n_params": n_params, "labels": label_ids(labels),
                        "train_records": len(X_train), "val_records": len(X_val),
                        "config": config}, fh, sort_keys=False)
    log(f"{model_name}: {n_params:,} parameters, train {len(X_train)} / val {len(X_val)} records, "
        f"device {device}, threads {torch.get_num_threads()}")

    pos_weight = None
    if dcfg.get("pos_weight", False):
        n_pos = np.clip(Y_train.sum(axis=0), 1, None)
        pos_weight = torch.tensor((len(Y_train) - n_pos) / n_pos, dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    batch_size = dcfg["batch_size"]
    steps_per_epoch = int(np.ceil(len(X_train) / batch_size))
    optimizer = torch.optim.AdamW(model.parameters(), lr=dcfg["lr"], weight_decay=dcfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=dcfg["lr"], epochs=epochs,
                                                    steps_per_epoch=steps_per_epoch)
    rng = np.random.default_rng(seed)
    state = {"epoch": 0, "best_metric": -np.inf, "best_epoch": 0, "bad_epochs": 0}

    last_path, best_path, hist_path = run_dir / "last.pt", run_dir / "best.pt", run_dir / "history.csv"
    if force:
        for p in (last_path, best_path, hist_path):
            p.unlink(missing_ok=True)
    elif last_path.exists():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        rng.bit_generator.state = ckpt["rng"]
        torch.set_rng_state(ckpt["torch_rng"])
        state = ckpt["state"]
        log(f"resumed from epoch {state['epoch']}")

    epoch_times, eval_times = [], []
    while state["epoch"] < epochs and state["bad_epochs"] < dcfg["patience"]:
        epoch = state["epoch"] + 1
        model.train()
        t0 = time.time()
        order = rng.permutation(len(X_train))
        total = 0.0
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            xb = np.stack([random_crop(X_train[i], window, rng) for i in idx])
            xb = torch.from_numpy(xb).to(device)
            yb = torch.from_numpy(Y_train[idx]).to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(idx)
        train_loss = total / len(order)
        epoch_time = time.time() - t0

        t0 = time.time()
        logits = predict_records(model, X_val, window, overlap, dcfg["eval_batch_size"], device)
        val_loss = float(criterion(torch.from_numpy(logits).to(device),
                                   torch.from_numpy(Y_val.astype(np.float32)).to(device)))
        metric = macro_summary(per_label_metrics(Y_val, _sigmoid(logits), labels))["macro_auroc_primary"]
        eval_time = time.time() - t0
        epoch_times.append(epoch_time)
        eval_times.append(eval_time)

        state["epoch"] = epoch
        if metric > state["best_metric"]:
            state.update(best_metric=metric, best_epoch=epoch, bad_epochs=0)
            torch.save({"model": model.state_dict(), "epoch": epoch, "metric": metric,
                        "model_name": model_name, "model_kwargs": model_kwargs}, best_path)
        else:
            state["bad_epochs"] += 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(), "state": state}, last_path)

        row = {"epoch": epoch, "train_loss": round(train_loss, 5), "val_loss": round(val_loss, 5),
               "val_macro_auroc": round(metric, 4), "epoch_time_s": round(epoch_time, 1),
               "eval_time_s": round(eval_time, 1), "lr": scheduler.get_last_lr()[0]}
        new = not hist_path.exists()
        with open(hist_path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        log(f"  epoch {epoch}: {row}")

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    t0 = time.time()
    logits = predict_records(model, X_val, window, overlap, dcfg["eval_batch_size"], device)
    infer_ms = (time.time() - t0) * 1000 / len(X_val)
    scores = _sigmoid(logits)
    return {"model": model, "scores": scores, "Y_val": Y_val, "val_meta": val.meta,
            "n_params": n_params, "best_epoch": best["epoch"], "epochs_run": state["epoch"],
            "train_time_s": float(np.sum(epoch_times)) if epoch_times else float("nan"),
            "mean_epoch_time_s": float(np.mean(epoch_times)) if epoch_times else float("nan"),
            "mean_eval_time_s": float(np.mean(eval_times)) if eval_times else float("nan"),
            "inference_ms_per_record": infer_ms, "model_path": best_path,
            "train_records": len(X_train)}


def evaluate_deep(result, labels):
    return evaluate(result["Y_val"], result["scores"], labels,
                    result["val_meta"]["patient_id"].values, n_boot=1000, seed=0)
