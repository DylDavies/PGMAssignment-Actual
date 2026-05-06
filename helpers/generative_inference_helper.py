import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
import config

def run_generative_inference(
    normal_train_df: pd.DataFrame, 
    test_df: pd.DataFrame, 
    learned_dags: dict[str, BayesianNetwork],
    threshold_perc: int = config.THRESHOLD_PERC
) -> pd.DataFrame:
    results_df = pd.DataFrame(index=test_df.index)
    
    for stage, dag in learned_dags.items():
        print(f"Starting {stage}...")
        sensors = list(dag.nodes())
        
        # Train on Normal data ONLY (No State Node)
        bn = BayesianNetwork(dag.edges())
        bn.add_nodes_from(sensors)
        
        all_states: dict[str, list[int]] = {s: list(range(config.N_BINS)) for s in sensors}
        
        bn.fit(
            normal_train_df[sensors], 
            estimator=BayesianEstimator, 
            prior_type="BDeu", 
            equivalent_sample_size=10,
            state_names=all_states
        )
        
        # Ultra-Fast Log-Likelihood Calculation via NumPy Indexing
        stage_log_likelihood = np.zeros(len(test_df))
        
        for cpd in bn.get_cpds(): # type: ignore
            node = cpd.variable
            parents = cpd.get_evidence()
            
            # cpd.values is an N-dimensional array. 
            # 1st dimension = the node's state. Subsequent dimensions = parents' states.
            # We build a tuple of arrays to extract the exact probability for every row instantly.
            indices = [test_df[node].values]
            for p in parents:
                indices.append(test_df[p].values)
                
            node_probs = cpd.values[tuple(indices)]
            
            stage_log_likelihood += np.log(node_probs + 1e-12)

        # First, calculate log-likelihoods for the training set to find the threshold
        train_ll = np.zeros(len(normal_train_df))
        for cpd in bn.get_cpds(): # type: ignore
            node = cpd.variable
            parents = cpd.get_evidence()
            
            indices = [normal_train_df[node].values]
            for p in parents:
                indices.append(normal_train_df[p].values)
                
            train_probs = cpd.values[tuple(indices)]
            train_ll += np.log(train_probs + 1e-12)
            
        threshold = np.percentile(train_ll, threshold_perc) 
        
        # Flag attacks if the test likelihood drops below the training threshold
        results_df[f"{stage}_Prediction"] = np.where(
            stage_log_likelihood < threshold, 
            config.ATTACK_LABEL, 
            config.NORMAL_LABEL
        )

        print(f" -> {stage} Complete.")
        
    return results_df