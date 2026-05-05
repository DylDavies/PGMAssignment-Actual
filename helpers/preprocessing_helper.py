import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from typing import Literal

import config

continuous_cols = []
discrete_cols = []

for stage in config.CONTINUOUS:
    continuous_cols.extend(config.CONTINUOUS[stage])
    discrete_cols.extend(config.DISCRETE[stage])

def impute(
    df: pd.DataFrame
) -> tuple[SimpleImputer, SimpleImputer]:
    # Continuous imputing
    continuous_imputer = SimpleImputer(strategy="median")
    df[continuous_cols] = continuous_imputer.fit_transform(df[continuous_cols])
    
    # Discrete Imputing
    discrete_imputer = SimpleImputer(strategy="most_frequent")
    df[discrete_cols] = discrete_imputer.fit_transform(df[discrete_cols])

    return continuous_imputer, discrete_imputer

def apply_imputer(
    df: pd.DataFrame,
    continuous_imputer: SimpleImputer,
    discrete_imputer: SimpleImputer
) -> None:
    df[continuous_cols] = continuous_imputer.transform(df[continuous_cols]) # type: ignore
    df[discrete_cols] = discrete_imputer.transform(df[discrete_cols]) # type: ignore

def scale(
    df: pd.DataFrame
) -> StandardScaler:
    sc = StandardScaler()
    df[continuous_cols] = sc.fit_transform(df[continuous_cols])

    return sc

def apply_scaler(
    df: pd.DataFrame,
    scaler: StandardScaler
) -> None:
    df[continuous_cols] = scaler.transform(df[continuous_cols])

def discretise(
    df: pd.DataFrame,
    n_bins: int = config.N_BINS,
    strategy: Literal['uniform', 'quantile', 'kmeans'] = config.DISCRETISE_STRATEGY
) -> tuple[KBinsDiscretizer, list[str]]:
    
    # Identify zero-variance continuous columns
    constant_cols = [col for col in continuous_cols if df[col].nunique() <= 1]
    
    if constant_cols:
        print(f"Dropping constant continuous sensors: {constant_cols}")
        df.drop(columns=constant_cols, inplace=True)
        
        # Remove from tracking list (modifying in place)
        for col in constant_cols:
            continuous_cols.remove(col)
            
        # CRITICAL: Remove from config so structure learning doesn't crash
        for stage in config.STAGE_SENSORS:
            config.STAGE_SENSORS[stage] = [
                s for s in config.STAGE_SENSORS[stage] if s not in constant_cols
            ]

    # Initialize Discretizer
    disc = KBinsDiscretizer(
        n_bins=n_bins, encode="ordinal", strategy=strategy, 
        subsample=200000, random_state=config.RANDOM_SEED
    )

    if continuous_cols: # Safety check in case all columns are dropped
        df[continuous_cols] = disc.fit_transform(df[continuous_cols]).astype(int)

    # Return the discretizer AND the list of dropped columns
    return disc, constant_cols

def apply_discretiser(
    df: pd.DataFrame,
    disc: KBinsDiscretizer,
    dropped_cols: list[str]
) -> None:
    # Drop the constant columns from train/test splits so shapes match
    if dropped_cols:
        cols_to_drop = [c for c in dropped_cols if c in df.columns]
        df.drop(columns=cols_to_drop, inplace=True)

    if not continuous_cols:
        return

    # Apply clipping to avoid out-of-bounds errors on test data
    for col, edges in zip(continuous_cols, disc.bin_edges_):
        lo, hi = edges[0], edges[-1]
        df[col] = df[col].clip(lower=lo, upper=hi)
        
    df[continuous_cols] = disc.transform(df[continuous_cols]).astype(int) # type: ignore

def coerce_discrete(
    df: pd.DataFrame
) -> None:
    for col in discrete_cols:
        try:
            df[col] = df[col].astype(float).astype(int)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Cannot coerce column '{col}' to int: {exc}") from exc
