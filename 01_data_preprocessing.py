import sys
from pathlib import Path

from config import ROOT_DIR
from helpers.data_helper import download_dataset, load_merged, split_train_test
from helpers.preprocessing_helper import impute, scale, apply_scaler

# Make sure dataset is downloaded and present
download_dataset()

print("Data present")

# Load merged DF
df = load_merged()

# Impute
impute(df)

# Create training and test split
train_df, test_df = split_train_test(df)

print(f"\nTrain: {len(train_df):,}")
print(f"Test : {len(test_df):,}")

# Scale continuous values to be betwee 0 and 1
scaler = scale(train_df)
apply_scaler(test_df, scaler)