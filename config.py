from pathlib import Path
from typing import Literal

RANDOM_SEED = 2026

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
CACHE_DIR = OUTPUT_DIR / "cache"

for dir in (DATA_DIR, OUTPUT_DIR, CACHE_DIR):
    dir.mkdir(parents=True, exist_ok=True)

KAGGLE_DATASET = "vishala28/swat-dataset-secure-water-treatment-system"

NORMAL_CSV = "normal.csv"
ATTACK_CSV = "attack.csv"
MERGED_CSV = "merged.csv"

LABEL_COL     = "Normal/Attack"
TIMESTAMP_COL = "Timestamp"
NORMAL_LABEL  = "Normal"
ATTACK_LABEL  = "Attack"

TRAIN_FRACTION = 0.8

STAGE_SENSORS: dict[str, list[str]] = {
    "P1": ["FIT101", "LIT101", "MV101", "P101", "P102"],
    "P2": ["AIT201", "AIT202", "AIT203", "FIT201", "MV201",
           "P201", "P202", "P203", "P204", "P205", "P206"],
    "P3": ["DPIT301", "FIT301", "LIT301", "MV301", "MV302",
           "MV303", "MV304", "P301", "P302"],
    "P4": ["AIT401", "AIT402", "FIT401", "LIT401", "P401", "P402",
           "P403", "P404", "UV401"],
    "P5": ["AIT501", "AIT502", "AIT503", "AIT504",
           "FIT501", "FIT502", "FIT503", "FIT504",
           "P501", "P502", "PIT501", "PIT502", "PIT503"],
    "P6": ["FIT601", "P601", "P602", "P603"],
}

# Sensors whose values are already integer-valued in the raw CSV
DISCRETE: dict[str, list[str]] = {
    "P1": ["MV101", "P101", "P102"],
    "P2": ["MV201", "P201", "P202", "P203", "P204", "P205", "P206"],
    "P3": ["MV301", "MV302", "MV303", "MV304", "P301", "P302"],
    "P4": ["P401", "P402", "P403", "P404", "UV401"],
    "P5": ["P501", "P502"],
    "P6": ["P601", "P602", "P603"],
}

# Sensors requiring discretisation (continuous float readings)
CONTINUOUS: dict[str, list[str]] = {
    stage: [s for s in sensors if s not in DISCRETE.get(stage, [])]
    for stage, sensors in STAGE_SENSORS.items()
}

N_BINS = 4
DISCRETISE_STRATEGY: Literal['uniform', 'quantile', 'kmeans'] = "uniform"