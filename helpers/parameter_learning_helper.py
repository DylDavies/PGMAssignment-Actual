import pandas as pd
import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
import config
import matplotlib.pyplot as plt
import networkx as nx
from joblib import Memory

memory = Memory(location=str(config.CACHE_DIR), verbose=0)

@memory.cache
def build_emission_models(
    train_df: pd.DataFrame, 
    learned_dags: dict[str, BayesianNetwork]
) -> dict[str, BayesianNetwork]:
    emission_models = {}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (stage, dag) in enumerate(learned_dags.items()):
        sensors = list(dag.nodes())
        
        # Create the new edges: Keep physical edges, add State -> Sensor edges
        new_edges = list(dag.edges())
        for sensor in sensors:
            new_edges.append(('State', sensor))
            
        # Instantiate the unified network
        bn = BayesianNetwork(new_edges)
        bn.add_nodes_from(sensors + ['State'])

        ax = axes[idx]
        
        pos = nx.circular_layout(bn)

        nx.draw(bn, pos, with_labels=True, node_color='skyblue', node_size=1500, arrowsize=20, ax=ax)
        
        ax.set_title(f"Learned DAG: {stage}", fontsize=11)
        ax.axis("off")
        ax.margins(0.2)
        
        # Prepare the training data (rename the label column to match our node)
        stage_df = train_df[sensors + [config.LABEL_COL]].copy()
        stage_df.rename(columns={config.LABEL_COL: 'State'}, inplace=True)

        all_states: dict[str, list[int | str]] = {sensor: list(range(config.N_BINS)) for sensor in sensors}
        all_states['State'] = [config.NORMAL_LABEL, config.ATTACK_LABEL]
        
        # Fit using Bayesian Estimation to prevent 0.0 probabilities on rare attacks
        bn.fit(
            stage_df, 
            estimator=BayesianEstimator, 
            prior_type="BDeu", 
            equivalent_sample_size=10,
            state_names=all_states
        )
        
        emission_models[stage] = bn

    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "EMPlots.png")

    return emission_models


def learn_transition_matrix(train_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    labels = train_df[config.LABEL_COL].values
    
    transitions = {
        config.NORMAL_LABEL: {config.NORMAL_LABEL: 0, config.ATTACK_LABEL: 0},
        config.ATTACK_LABEL: {config.NORMAL_LABEL: 0, config.ATTACK_LABEL: 0}
    }
    
    # Count every state transition from t-1 to t
    for i in range(1, len(labels)):
        prev_state = labels[i-1]
        curr_state = labels[i]
        transitions[prev_state][curr_state] += 1
        
    # Convert counts to probabilities (normalize rows)
    transition_probs = {}
    for state, counts in transitions.items():
        total = sum(counts.values())
        transition_probs[state] = {
            config.NORMAL_LABEL: counts[config.NORMAL_LABEL] / total,
            config.ATTACK_LABEL: counts[config.ATTACK_LABEL] / total
        }
    
    return transition_probs