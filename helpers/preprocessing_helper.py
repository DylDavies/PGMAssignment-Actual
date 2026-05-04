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
) -> KBinsDiscretizer:
    disc = KBinsDiscretizer(
        n_bins=n_bins, encode="ordinal", strategy=strategy, subsample=None
    )

    df[continuous_cols] = disc.fit_transform(df[continuous_cols]).astype(int)

    return disc

def apply_discretiser(
    df: pd.DataFrame,
    disc: KBinsDiscretizer,
) -> pd.DataFrame:
    for col, edges in zip(continuous_cols, disc.bin_edges_):
        lo, hi = edges[0], edges[-1]
        df[col] = df[col].clip(lower=lo, upper=hi)
    df[continuous_cols] = disc.transform(df[continuous_cols]).astype(int)
    return df