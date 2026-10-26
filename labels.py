"""
The rhythm labels the product reports, and which PTB-XL codes map to each one.

DRAFT for the MVP. Ali: check the number of PTB-XL records per label before training.
SVT and high-grade AV block have very few examples in PTB-XL (a few dozen each), so they may
need to be merged, dropped from the model, or reported as a limitation.
"""

LABELS = [
    {"id": "NSR",      "name": "Normal sinus rhythm",              "ptbxl_codes": ["SR"]},
    {"id": "SBRAD",    "name": "Sinus bradycardia",                "ptbxl_codes": ["SBRAD"]},
    {"id": "STACH",    "name": "Sinus tachycardia",                "ptbxl_codes": ["STACH"]},
    {"id": "AFIB",     "name": "Atrial fibrillation",              "ptbxl_codes": ["AFIB"]},
    {"id": "AFLT",     "name": "Atrial flutter",                   "ptbxl_codes": ["AFLT"]},
    {"id": "SVT",      "name": "Supraventricular tachycardia",     "ptbxl_codes": ["SVTAC", "PSVT"]},
    {"id": "AVB1",     "name": "First-degree AV block",            "ptbxl_codes": ["1AVB"]},
    {"id": "AVB_HIGH", "name": "Second/third-degree AV block",     "ptbxl_codes": ["2AVB", "3AVB"]},
    {"id": "LBBB",     "name": "Left bundle branch block",         "ptbxl_codes": ["CLBBB"]},
    {"id": "RBBB",     "name": "Right bundle branch block",        "ptbxl_codes": ["CRBBB"]},
    {"id": "PVC",      "name": "Premature ventricular complexes",  "ptbxl_codes": ["PVC"]},
    {"id": "PAC",      "name": "Premature atrial complexes",       "ptbxl_codes": ["PAC"]},
]

LABEL_IDS = [l["id"] for l in LABELS]
LABEL_NAMES = {l["id"]: l["name"] for l in LABELS}
LABEL_NAMES["UNCERTAIN"] = "Uncertain - needs expert review"

# Symptoms the frontend can send in the patient context
SYMPTOMS = ["chest_pain", "dyspnea", "palpitations", "syncope", "dizziness",
            "altered_mental_status", "shock", "heart_failure"]
