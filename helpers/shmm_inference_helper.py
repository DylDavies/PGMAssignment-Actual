import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from joblib import Memory
import config

memory = Memory(location=str(config.CACHE_DIR), verbose=0)

@memory.cache
def run_shmm_inference(
    test_df: pd.DataFrame, 
    emission_models: dict[str, BayesianNetwork],
    transition_matrix: dict[str, dict[str, float]],
    train_df: pd.DataFrame
) -> pd.DataFrame:
    p_train_normal = (train_df[config.LABEL_COL] == config.NORMAL_LABEL).mean()
    p_train_attack = (train_df[config.LABEL_COL] == config.ATTACK_LABEL).mean()

    results_df = pd.DataFrame(index=test_df.index)
    
    for stage, bn in emission_models.items():
        print(f"Starting {stage}...")
        sensors = [node for node in bn.nodes() if node != 'State']
        
        static_probs = bn.predict_probability(test_df[sensors])
        
        col_normal = f"State_{config.NORMAL_LABEL}"
        col_attack = f"State_{config.ATTACK_LABEL}"
        
        emission_normal = static_probs[col_normal].values / p_train_normal # type: ignore
        emission_attack = static_probs[col_attack].values / p_train_attack # type: ignore
        
        n_steps = len(test_df)
        belief_normal = np.zeros(n_steps)
        belief_attack = np.zeros(n_steps)
        
        # Initialize t=0
        belief_normal[0] = emission_normal[0] * p_train_normal
        belief_attack[0] = emission_attack[0] * p_train_attack
        
        # Normalize t=0
        norm = belief_normal[0] + belief_attack[0]
        belief_normal[0] /= norm
        belief_attack[0] /= norm
        
        # Extract transition constants for maximum loop speed
        t_nn = transition_matrix[config.NORMAL_LABEL][config.NORMAL_LABEL]
        t_na = transition_matrix[config.NORMAL_LABEL][config.ATTACK_LABEL]
        t_an = transition_matrix[config.ATTACK_LABEL][config.NORMAL_LABEL]
        t_aa = transition_matrix[config.ATTACK_LABEL][config.ATTACK_LABEL]
        
        for t in range(1, n_steps):
            # Prior for time t based on the transitioned belief from t-1
            prior_n = (belief_normal[t-1] * t_nn) + (belief_attack[t-1] * t_an)
            prior_a = (belief_normal[t-1] * t_na) + (belief_attack[t-1] * t_aa)
            
            # Update with current emission evidence
            unnorm_n = emission_normal[t] * prior_n
            unnorm_a = emission_attack[t] * prior_a
            
            # Normalize to prevent underflow
            norm = unnorm_n + unnorm_a
            if norm == 0: # Failsafe for extreme floating point precision limits
                norm = 1e-12
                
            belief_normal[t] = unnorm_n / norm
            belief_attack[t] = unnorm_a / norm
            
        # Store results
        results_df[f"{stage}_Prob_Attack"] = belief_attack
        results_df[f"{stage}_Prediction"] = np.where(belief_attack > 0.5, config.ATTACK_LABEL, config.NORMAL_LABEL)
        print(f" -> {stage} Complete.")

    return results_df