from time import perf_counter
import pandas as pd

from helpers.data_helper import download_dataset, load_merged, split_train_test
from helpers.preprocessing_helper import impute, apply_imputer, discretise, apply_discretiser, coerce_discrete
from helpers.structure_learning_helper import learn_all_structures
from helpers.parameter_learning_helper import build_emission_models, learn_transition_matrix
from helpers.shmm_inference_helper import run_shmm_inference
from helpers.evaluation_helper import evaluate_system
from helpers.generative_inference_helper import run_generative_inference
from helpers.hmm_inference_helper import run_hmm_inference

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
hmm_results: pd.DataFrame | None = None

if 'goc' in config.MODELS_TO_RUN or 'shmm' in config.MODELS_TO_RUN:
    shmm_start = perf_counter()
    generative_start = perf_counter()

    # Structure Learning
    learned_networks = learn_all_structures(normal_train)

    if 'shmm' in config.MODELS_TO_RUN:
        # Build the Emission Models (P(Sensors_t | State_t))
        emission_models = build_emission_models(train_df, learned_networks)

        # Build the Transition Model (P(State_t | State_t-1))
        transition_matrix = learn_transition_matrix(train_df)

        # Testing
        print("Running SHMM Inference...")
        shmm_results = run_shmm_inference(test_df, emission_models, transition_matrix, train_df)
        shmm_end = perf_counter()

        print(f"SHMM Inference Execution Time: {(shmm_end - shmm_start):.4f}")

    print("Running Generative Inference...")
    generative_results = run_generative_inference(normal_train, test_df, learned_networks)
    generative_end = perf_counter()

    print(f"Generative Inference Execution Time: {(generative_end - generative_start):.4f}") # this will only be accurage if shmm does not run due to code structure

if 'hmm' in config.MODELS_TO_RUN:
    hmm_start = perf_counter()
    print("Running HMM Inference...")
    hmm_results = run_hmm_inference(normal_train, test_df)
    hmm_end = perf_counter()

    print(f"HMM Inference Execution Time: {(hmm_end - hmm_start):.4f}")

# Evaluation

if shmm_results is not None:
    print("\n--- SHMM Evaluation ---")
    evaluate_system(shmm_results, test_df, "confusion_matrix_shmm")

if generative_results is not None:
    print("\n--- Generative Evaluation ---")
    evaluate_system(generative_results, test_df, "confusion_matrix_generative")

if hmm_results is not None:
    print("\n--- HMM Evaluation ---")
    evaluate_system(hmm_results, test_df, "confusion_matrix_hmm")