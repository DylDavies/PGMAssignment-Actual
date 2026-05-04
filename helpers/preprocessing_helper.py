import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler

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