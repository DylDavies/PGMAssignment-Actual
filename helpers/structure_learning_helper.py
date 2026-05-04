import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.estimators import BicScore, HillClimbSearch
from pgmpy.base import DAG
import pandas as pd

from config import STAGE_SENSORS

def learn_all_structures(normal_train_df: pd.DataFrame) -> dict[str, DAG]:
    learned_dags = {}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (stage, sensors) in enumerate(STAGE_SENSORS.items()):
        print(f"Learning structure for {stage} ({len(sensors)} sensors)...")
        
        stage_df = normal_train_df[sensors]
        
        hc = HillClimbSearch(stage_df)
        dag = DAG(hc.estimate(scoring_method=BicScore(stage_df), show_progress=True))
        
        learned_dags[stage] = dag
        
        print(f" -> Found {len(dag.edges())} edges.")
        
        ax = axes[idx]
        nx.draw_networkx(
            dag,
            ax=ax,
            with_labels=True,
            node_color='lightblue',
            node_size=800,
            font_size=8,
            arrows=True,
            connectionstyle="arc3,rad=0.1"
        )
        ax.set_title(f"Learned DAG: {stage}")

    plt.tight_layout()
    plt.show()
    
    return learned_dags