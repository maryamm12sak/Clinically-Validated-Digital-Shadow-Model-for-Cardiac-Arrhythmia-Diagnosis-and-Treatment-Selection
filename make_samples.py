"""
Downloads demo ECGs from PTB-XL into the samples/ folder and writes samples/index.json.
Run:  python make_samples.py   (needs internet)

- Needs ptbxl_database.csv in this folder (or downloads it, which can be slow).
- Records come from fold 10 (the test fold), so the model never trained on them.
- Safe to re-run: samples already saved are skipped, failed ones are retried.
"""
import ast
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import wfdb

BASE = "https://physionet.org/files/ptb-xl/1.0.3/"
HERE = Path(__file__).parent
OUT = HERE / "samples"
RAW = OUT / "raw"
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
USE_500HZ = False     # 100 Hz files are 5x smaller; set True if downloads are fast

# id, description, PTB-XL code that must be present
TARGETS = [
    ("normal",      "Normal sinus rhythm",                "NORM"),
    ("sinus_brady", "Sinus bradycardia",                  "SBRAD"),
    ("sinus_tachy", "Sinus tachycardia",                  "STACH"),
    ("afib",        "Atrial fibrillation",                "AFIB"),
    ("aflutter",    "Atrial flutter",                     "AFLT"),
    ("avb1",        "First-degree AV block",              "1AVB"),
    ("avb3",        "Third-degree (complete) AV block",   "3AVB"),
    ("lbbb",        "Left bundle branch block",           "CLBBB"),
    ("rbbb",        "Right bundle branch block",          "CRBBB"),
    ("pvc",         "Premature ventricular complexes",    "PVC"),
]


def download(url, dest, tries=3, timeout=30):
    """Download one file with a time limit and retries. Returns True on success."""
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            dest.write_bytes(r.content)
            return True
        except Exception as e:
            print(f"      attempt {attempt} failed ({type(e).__name__}), retrying..." if attempt < tries
                  else f"      gave up ({type(e).__name__})")
            time.sleep(2)
    return False


# ---- metadata ----
local_csv = HERE / "ptbxl_database.csv"
if local_csv.exists():
    print("Using local ptbxl_database.csv")
    db = pd.read_csv(local_csv, index_col="ecg_id")
else:
    print("Downloading PTB-XL metadata (slow; or save ptbxl_database.csv from your browser into this folder)...")
    db = pd.read_csv(BASE + "ptbxl_database.csv", index_col="ecg_id")
db = db[db["scp_codes"].astype(str).str.startswith("{")]   # skip any incomplete rows
db["codes"] = db["scp_codes"].apply(ast.literal_eval)
test = db[db["strat_fold"] == 10]

OUT.mkdir(exist_ok=True)
RAW.mkdir(exist_ok=True)
index_file = OUT / "index.json"
index = {s["id"]: s for s in json.loads(index_file.read_text(encoding="utf-8"))} if index_file.exists() else {}

# ---- records ----
for sample_id, description, code in TARGETS:
    if sample_id in index and (OUT / index[sample_id]["file"]).exists():
        print(f"  {sample_id}: already saved, skipping")
        continue
    if code == "NORM":   # cleanest normal: only NORM + sinus rhythm
        matches = test[test["codes"].apply(lambda c: set(c) == {"NORM", "SR"})]
    else:
        matches = test[test["codes"].apply(lambda c: code in c)]
    if matches.empty:
        print(f"  {sample_id}: no example found, skipped")
        continue
    ecg_id = matches["codes"].apply(len).sort_values(kind="stable").index[0]   # fewest other labels
    row = db.loc[ecg_id]
    path = row["filename_hr"] if USE_500HZ else row["filename_lr"]          # e.g. records100/00000/00009_lr
    name = path.rsplit("/", 1)[1]

    print(f"  {sample_id}: downloading PTB-XL record {ecg_id}...")
    ok = all(download(BASE + path + ext, RAW / (name + ext)) for ext in (".hea", ".dat"))
    if not ok:
        print(f"  {sample_id}: FAILED, run the script again later to retry")
        continue

    rec = wfdb.rdrecord(str(RAW / name))
    filename = f"{sample_id}.csv"
    np.savetxt(OUT / filename, rec.p_signal, delimiter=",", fmt="%.4f", header=",".join(LEADS), comments="")
    index[sample_id] = {"id": sample_id, "description": description, "file": filename,
                        "source": f"PTB-XL ecg_id {ecg_id}", "ptbxl_codes": sorted(row["codes"]),
                        "fs": int(rec.fs),
                        "age": None if pd.isna(row["age"]) else ("90+" if row["age"] >= 300 else int(row["age"])),   # PTB-XL stores 90+ as 300
                        "sex": "male" if row["sex"] == 0 else "female"}
    index_file.write_text(json.dumps(list(index.values()), indent=2), encoding="utf-8")
    print(f"  {sample_id}: saved {filename}  {sorted(row['codes'])}")

print(f"Done: {len(index)} of {len(TARGETS)} samples in {OUT}")
