from time import perf_counter
import pandas as pd

from helpers.data_helper import download_dataset, load_merged, split_train_test
from helpers.preprocessing_helper import impute, apply_imputer, discretise, apply_discretiser, coerce_discrete
from helpers.structure_learning_helper import learn_all_structures
from helpers.parameter_learning_helper import build_emission_models, learn_transition_matrix
from helpers.shmm_inference_helper import run_hmm_inference
from helpers.evaluation_helper import evaluate_system
from helpers.generative_inference_helper import run_generative_inference

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

# Discretise
discretiser, dropped_cols = discretise(normal_train)
apply_discretiser(train_df, discretiser, dropped_cols)
apply_discretiser(test_df, discretiser, dropped_cols)

coerce_discrete(normal_train)
coerce_discrete(train_df)
coerce_discrete(test_df)

shmm_results: pd.DataFrame | None = None
generative_results: pd.DataFrame | None = None

if 'goc' in config.MODELS_TO_RUN or 'shmm' in config.MODELS_TO_RUN:
    # Structure Learning
    learned_networks = learn_all_structures(normal_train)

    if 'shmm' in config.MODELS_TO_RUN:
        # Build the Emission Models (P(Sensors_t | State_t))
        emission_models = build_emission_models(train_df, learned_networks)

        # Build the Transition Model (P(State_t | State_t-1))
        transition_matrix = learn_transition_matrix(train_df)

        # Testing
        shmm_start = perf_counter()
        print("Running HMM Inference...")
        shmm_results = run_hmm_inference(test_df, emission_models, transition_matrix, train_df)
        shmm_end = perf_counter()

        print(f"HMM Inference Execution Time: {(shmm_end - shmm_start):.4f}")

    generative_start = perf_counter()
    print("Running Generative Inference...")
    generative_results = run_generative_inference(normal_train, test_df, learned_networks)
    generative_end = perf_counter()

    print(f"Generative Inference Execution Time: {(generative_end - generative_start):.4f}")

# Evaluation

if shmm_results is not None:
    print("\n--- SHMM Evaluation ---")
    evaluate_system(shmm_results, test_df, "confusion_matrix_hmm")

if generative_results is not None:
    print("\n--- Generative Evaluation ---")
    evaluate_system(generative_results, test_df, "confusion_matrix_generative")