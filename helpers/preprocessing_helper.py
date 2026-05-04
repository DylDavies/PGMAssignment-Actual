import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler

import config

continuous_cols = []
discrete_cols = []

for stage in config.CONTINUOUS:
    continuous_cols.extend(config.CONTINUOUS[stage])
    discrete_cols.extend(config.DISCRETE[stage])

def impute(
    df: pd.DataFrame
) -> None:
    # Continuous imputing
    continuous_imputer = SimpleImputer(strategy="median")
    df[continuous_cols] = continuous_imputer.fit_transform(df[continuous_cols])

    
    # Discrete Imputing
    discrete_imputer = SimpleImputer(strategy="most_frequent")
    df[discrete_cols] = discrete_imputer.fit_transform(df[discrete_cols])

def scale(
    df: pd.DataFrame
) -> MinMaxScaler:
    sc = MinMaxScaler()
    df[continuous_cols] = sc.fit_transform(df[continuous_cols])

    return sc

def apply_scaler(
    df: pd.DataFrame,
    scaler: MinMaxScaler
) -> None:
    df[continuous_cols] = scaler.transform(df[continuous_cols])