import numpy as np
from pathlib import Path
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

import config

def download_dataset(dest: Path = config.DATA_DIR) -> None:

    target = dest / config.NORMAL_CSV

    if target.exists():
        return
    
    api = KaggleApi()
    api.authenticate()

    api.dataset_download_files(
        config.KAGGLE_DATASET, path=str(dest), unzip=True, quiet=False
    )

def load(file: Path) -> pd.DataFrame:
    df = pd.read_csv(file)
    
    df.columns = df.columns.str.strip()
    df[config.TIMESTAMP_COL] = pd.to_datetime(df[config.TIMESTAMP_COL], errors="coerce", format="%d/%m/%Y %I:%M:%S %p")
    df[config.LABEL_COL] = df[config.LABEL_COL].astype(str).str.strip()

    df = df.sort_values(config.TIMESTAMP_COL).reset_index(drop=True)
    
    return df

def load_normal() -> pd.DataFrame:
    return load(config.DATA_DIR / config.NORMAL_CSV)

def load_attack() -> pd.DataFrame:
    return load(config.DATA_DIR / config.ATTACK_CSV)

def load_merged() -> pd.DataFrame:
    return load(config.DATA_DIR / config.MERGED_CSV)

def split_train_test(
    df: pd.DataFrame,
    frac: float = config.TRAIN_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_train = int(len(df) * frac)

    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()