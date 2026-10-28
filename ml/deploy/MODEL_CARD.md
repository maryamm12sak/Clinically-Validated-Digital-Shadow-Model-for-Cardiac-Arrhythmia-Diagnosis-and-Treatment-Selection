# Model card: ptbxl-arrhythmia-v1.0

## Intended use

Decision support for clinicians reading a standard 12-lead resting ECG: the model scores 12 rhythm and
conduction labels. **It is not a diagnosis.** The treating physician interprets the ECG and is
responsible for every decision. Not validated outside PTB-XL; not cleared for clinical use.

Shipped for the MVP: **`resnet_only`** (ResNet1d, 3 training seeds averaged).
Documented upgrade: **`ensemble`** (ResNet1d + LightGBM on hand-crafted 500 Hz features).

## Input / output contract

| | |
|---|---|
| Input | `sig`: float array `(samples, 12)` in **millivolts**, leads in the order `I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`; `fs`: sampling rate in Hz (any; resampled internally) |
| Length | 5 to 60 s (`ValueError` otherwise, and for NaN/Inf or a wrong number of leads) |
| `predict(sig, fs)` | `{label_id: calibrated_probability}` for all 12 labels, in `labels.yaml` order |
| `predict_detailed(sig, fs)` | per label: `resnet_probability`, `lightgbm_probability` (ensemble only), `raw_probability`, `calibrated_probability`, `threshold_max_f1`, `threshold_high_sensitivity`, `detected`, `detected_high_sensitivity`, `exploratory`, `note`; plus `mode`, `components`, `warnings`, `runtime_ms` |

What happens inside (`resnet_only`):
1. resample to 100 Hz; z-score each lead with the training-fold mean/std (`models/normalizer_seed*.json`);
2. cut 2.5 s windows (250 samples) with 50 % overlap covering the whole signal (last window ends at the last sample);
3. each ONNX model (`models/resnet1d_seed*.onnx`, input `[batch, 12, 250]` float32) gives **logits** `[windows, 12]`;
4. per label, take the **max logit over windows**, apply the sigmoid -> **raw probability** per seed; average the 3 seeds;
5. per-label **calibration** (none / temperature / Platt, chosen by cross-validation on PTB-XL fold 9) -> **calibrated probability**;
6. compare with the per-label **threshold** (fitted on fold 9) -> `detected`.

Terms: *logits* are unbounded model outputs; *raw probabilities* are their sigmoid (ranking is good,
values are not trustworthy as risks); *calibrated probabilities* are adjusted so that, on PTB-XL,
"0.3" means roughly 30 % of such ECGs carry the label; *thresholds* turn calibrated probabilities into
yes/no. Two threshold sets are provided: `max_f1` (per label, best F1 on fold 9) and
`high_sensitivity` (sensitivity >= 0.90 on fold 9 for AFIB, AFLT, SVT, AVB_HIGH, LBBB; max-F1 otherwise).

## Labels

| id | name | PTB-XL codes | status |
|---|---|---|---|
| NSR | Normal sinus rhythm | SR | primary |
| SBRAD | Sinus bradycardia | SBRAD | primary |
| STACH | Sinus tachycardia | STACH | primary |
| AFIB | Atrial fibrillation | AFIB | primary |
| AFLT | Atrial flutter | AFLT | **exploratory** |
| SVT | Supraventricular tachycardia | SVTAC, PSVT | **exploratory** |
| AVB1 | First-degree AV block | 1AVB | primary |
| AVB_HIGH | Second/third-degree AV block | 2AVB, 3AVB | **exploratory** |
| LBBB | Left bundle branch block | CLBBB | primary |
| RBBB | Right bundle branch block | CRBBB | primary |
| PVC | Premature ventricular complexes | PVC | primary |
| PAC | Premature atrial complexes | PAC | primary |

Exploratory labels have 3-7 positive ECGs in each of the validation and test folds: their numbers are
not reliable, and any detection must be shown as "needs expert review".

## Training and evaluation

- Data: PTB-XL v1.0.3 (21,799 ECGs, 18,869 patients; 10 s, 12-lead). Folds 1-8 training, fold 9
  validation (early stopping, calibration, thresholds), fold 10 test, evaluated **once** after
  pre-registration (commit 14486d8, SHA-256 of every model/calibrator/threshold file). No patient
  appears in more than one split.
- Label rule: a label is present when its SCP code appears in the record (likelihood ignored).
- ResNet1d: xresnet1d-style 1D CNN, 1.93 M parameters, 2.5 s random crops, AdamW + OneCycle
  (max lr 1e-2), early stopping on fold-9 macro-AUROC; 3 seeds.

## Fold-10 (test) results, 2,198 ECGs

Macro averages are over the 9 primary labels; 95 % CI from a patient-level bootstrap (1,000 resamples).

| | resnet_only (shipped) | ensemble (upgrade) |
|---|---|---|
| macro-AUROC | **0.9709 [0.9661-0.9755]** | 0.9768 [0.9727-0.9806] |
| macro-AUPRC | 0.786 | 0.805 |
| macro-F1 / micro-F1 (max-F1 thresholds) | 0.768 / 0.887 | 0.775 / 0.892 |
| mean ECE, primary labels (raw -> calibrated) | 0.027 -> 0.011 | 0.023 -> 0.012 |

Per label, `resnet_only`, max-F1 thresholds (AFIB and LBBB also with the high-sensitivity threshold):

| label | n_pos | AUROC | AUPRC | sensitivity | specificity |
|---|---|---|---|---|---|
| NSR | 1674 | 0.867 | 0.941 | 0.974 | 0.584 |
| SBRAD | 64 | 0.951 | 0.577 | 0.438 | 0.994 |
| STACH | 82 | 0.996 | 0.912 | 0.915 | 0.993 |
| AFIB | 152 | 0.983 | 0.932 | 0.882 (0.875 high-sens.) | 0.993 |
| AVB1 | 79 | 0.979 | 0.603 | 0.620 | 0.980 |
| LBBB | 54 | 0.996 | 0.894 | 0.907 (0.963 high-sens.) | 0.996 |
| RBBB | 54 | 0.998 | 0.932 | 0.981 | 0.993 |
| PVC | 114 | 0.991 | 0.794 | 0.939 | 0.984 |
| PAC | 40 | 0.976 | 0.494 | 0.600 | 0.987 |
| AFLT (exploratory) | 7 | 0.955 | 0.623 | 0.000 | 1.000 |
| SVT (exploratory) | 5 | 0.996 | 0.348 | 0.600 | 0.995 |
| AVB_HIGH (exploratory) | 3 | 0.996 | 0.593 | 0.667 | 1.000 |

All numbers are in `manifest.json` -> `test_metrics`.

## Known limitations and failure cases

- **NSR**: specificity only ~0.58-0.64 at the max-F1 threshold; many non-sinus or mixed ECGs get a high
  NSR probability. Do not treat "NSR" as "normal ECG".
- **SBRAD**: misses about half of the cases at the max-F1 threshold (sensitivity 0.44-0.52).
- **PAC and AVB1**: precision about 0.5 (one in two detections is a false alarm); AVB1 sensitivity ~0.62-0.70.
- **Exploratory labels (AFLT, SVT, AVB_HIGH)**: unreliable. The high-sensitivity AFLT threshold is
  effectively 0 (hundreds of false positives) and must not be used.
- **No ventricular tachycardia label** (PTB-XL has almost no VT): the model cannot flag VT.
- **Single dataset**: PTB-XL, recorded in Germany 1989-1996 on Schiller devices. No external validation;
  performance on other populations, devices and filters (e.g. Pakistani hospitals) is unknown.
- 10 s training ECGs; 5-60 s inputs are supported by windowing, but only 10 s was evaluated.
- In `ensemble` mode at sampling rates below 500 Hz the feature part runs on upsampled data (a warning
  is returned); only 500 Hz input matches training.

## Runtime and dependencies

- `resnet_only`: numpy, scipy, onnxruntime (tested with onnxruntime 1.31.1, numpy 2.5.4, scipy 1.18.1), pyyaml.
  About 15 ms per 10 s ECG on one CPU thread (single call, including resampling; Ryzen 7 5700X).
- `ensemble`: additionally lightgbm 4.7.0, neurokit2 0.2.13, PyWavelets 1.10.0; about 140 ms per ECG on one thread
  (the 500 Hz feature extraction dominates).
- Python 3.14 (backend): PyPI has Windows wheels for onnxruntime 1.31.1 (cp314), numpy, scipy and
  PyWavelets (cp314) and a Python-version-independent LightGBM 4.7.0 wheel; neurokit2 is pure Python.
  Checked on PyPI 2026-10-28; not test-installed on 3.14.
- Package size: about 42 MB (3 x 7.4 MB ONNX, 36 LightGBM text models, JSON).
