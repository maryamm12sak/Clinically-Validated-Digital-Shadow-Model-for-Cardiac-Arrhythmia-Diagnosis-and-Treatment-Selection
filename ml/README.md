# ML benchmark (PTB-XL)

Model development code, kept separate from the FastAPI backend at the repo root.

## Layout

```
ml/
  configs/
    labels.yaml        12 product labels -> PTB-XL SCP codes, primary/exploratory tiers
    benchmark.yaml     data paths, folds, seeds, preprocessing options
  src/
    data_loader.py     PTB-XL metadata, cached signals (.npy), validation report, splits
    labels.py          multi-hot label matrix, tier indices, per-split counts
    preprocessing.py   lead order, resampling, bandpass, normalizer, crops/windows
    feature_extraction.py  hand-crafted features for the classical baselines
    evaluate.py        AUROC/AUPRC, macro over primary labels, patient-level bootstrap CI
    models/classical.py    per-label logistic regression, random forest, extra trees, LightGBM
  benchmark.py         trains on folds 1-8, evaluates on fold 9, writes ml/results/
  tests/               pytest suite
  requirements-ml.txt
```

## Classical baselines

```
cd ml
python benchmark.py                                   # all runs
python benchmark.py --models logreg,lightgbm --dev-subset
python benchmark.py --force                           # rebuild features, redo finished runs
```

Features are extracted once at 500 Hz to `data.features_path` and reused. Results are appended to
`ml/results/benchmark_results.csv` and `ml/results/per_class_metrics.csv`; fold-9 predictions and
fitted models go to `ml/results/predictions/` and `ml/results/models/` (not committed). Finished
runs are skipped unless `--force`.

## Setup and tests

```
pip install -r ml/requirements-ml.txt --extra-index-url https://download.pytorch.org/whl/cpu
python -m pytest ml/tests -q
```

Set `data.data_root` and `data.cache_dir` in `configs/benchmark.yaml` for your machine.
The first test run builds the 100 Hz signal cache (about 1 GB) in `cache_dir`; later runs reuse it.

## Conventions

- **Splits:** folds 1-8 train, fold 9 validation, fold 10 test. Fold 10 is locked:
  `get_split("test")` raises unless called with `allow_test=True`, which is reserved for the
  final evaluation. No patient appears in more than one split (checked by
  `assert_no_patient_leakage`).
- **Label rule:** a label is positive when any of its SCP code keys is present in `scp_codes`,
  ignoring likelihood. PTB-XL stores most rhythm codes (SR, AFIB, ...) with likelihood 0.
  Merged labels (SVT = SVTAC or PSVT, AVB_HIGH = 2AVB or 3AVB) are a logical OR.
- **Tiers:** AFLT, SVT and AVB_HIGH are exploratory (too few records to validate) and excluded
  from macro-AUROC.
- **No silent cleaning:** invalid records (flat lead, |x| > 10 mV, NaN/Inf, wrong shape) are
  listed in the validation report with reason codes and only dropped if
  `data.exclude_invalid: true`.
- **One preprocessing path:** training and inference both call `preprocessing.preprocess()`;
  the normalizer is fitted on the train split only and saved as JSON.
