from helpers.data_helper import download_dataset, load_merged, split_train_test
from helpers.preprocessing_helper import impute, apply_imputer, discretise, apply_discretiser, coerce_discrete
from helpers.structure_learning_helper import learn_all_structures
from helpers.parameter_learning_helper import build_emission_models, learn_transition_matrix
from helpers.inference_helper import run_hmm_inference
from helpers.evaluation_helper import evaluate_system

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

# Structure Learning
learned_networks = learn_all_structures(normal_train)

# Build the Emission Models (P(Sensors_t | State_t))
emission_models = build_emission_models(train_df, learned_networks)

# Build the Transition Model (P(State_t | State_t-1))
transition_matrix = learn_transition_matrix(train_df)

# Testing
results = run_hmm_inference(test_df, emission_models, transition_matrix, train_df)

# Evaluation
evaluate_system(results, test_df)