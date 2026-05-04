import sys
from pathlib import Path

from config import ROOT_DIR
from helpers.data_helper import download_dataset, load_merged, split_train_test
from helpers.preprocessing_helper import impute, apply_imputer, scale, apply_scaler

import config

# Make sure dataset is downloaded and present
download_dataset()

# Load merged DF
df = load_merged()

# Create training and test split
train_df, test_df = split_train_test(df)
normal_train = train_df[train_df[config.LABEL_COL] == config.NORMAL_LABEL].copy()

print(f"\nTrain: {len(train_df):,}  |  Normal-train: {len(normal_train):,}")
print(f"Test : {len(test_df):,}")

# Impute
continuous_imputer, discrete_imputer = impute(normal_train)
apply_imputer(train_df, continuous_imputer, discrete_imputer)
apply_imputer(test_df, continuous_imputer, discrete_imputer)

# Scale continuous values to be between 0 and 1
scaler = scale(normal_train)
apply_scaler(train_df, scaler)
apply_scaler(test_df, scaler)