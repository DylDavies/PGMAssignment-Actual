import logging
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from joblib import Memory
from pgmpy.estimators import BicScore, HillClimbSearch
from pgmpy.models import BayesianNetwork

import config

log = logging.getLogger(__name__)
memory = Memory(location=str(config.CACHE_DIR), verbose=0)

def _check_integer_dtypes(df: pd.DataFrame) -> None:
    """Raise ValueError if any column is not integer-typed."""
    non_int = [c for c in df.columns if not pd.api.types.is_integer_dtype(df[c])]
    if non_int:
        raise ValueError(f"Structure learning requires integer-typed columns. Found: {non_int}")

@memory.cache
def _cached_hillclimb(df_stage: pd.DataFrame, max_indegree: int = 4) -> list[tuple[str, str]]:
    """Cached execution of the Hill Climb Search for a specific dataframe."""
    hc = HillClimbSearch(df_stage)
    scorer = BicScore(df_stage)
    
    best_model = hc.estimate(
        scoring_method=scorer, # type: ignore
        max_indegree=max_indegree,
        show_progress=False
    )
    return list(best_model.edges())

def to_bayesian_network(edges: list[tuple[str, str]], nodes: list[str]) -> BayesianNetwork:
    """Builds a BayesianNetwork, ensuring isolated nodes are not dropped."""
    bn = BayesianNetwork(edges)
    bn.add_nodes_from(nodes)
    return bn

def learn_all_structures(df_normal_disc: pd.DataFrame) -> dict[str, BayesianNetwork]:    
    learned_networks = {}
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (stage, sensors) in enumerate(config.STAGE_SENSORS.items()):
        print(f"Learning structure for {stage} ({len(sensors)} sensors)...")
        
        stage_df = df_normal_disc[sensors]
        _check_integer_dtypes(stage_df)
        
        edges = _cached_hillclimb(stage_df)
        print(f" -> Found {len(edges)} edges.")
        
        bn = to_bayesian_network(edges, sensors)
        learned_networks[stage] = bn
        
        ax = axes[idx]
        
        pos = nx.circular_layout(bn)

        nx.draw(bn, pos, with_labels=True, node_color='skyblue', node_size=1500, arrowsize=20, ax=ax)
        
        ax.set_title(f"Learned DAG: {stage}\n({len(edges)} edges)", fontsize=11)
        ax.axis("off")

    # plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / "BNPlots.png")
    
    return learned_networks