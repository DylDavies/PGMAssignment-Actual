import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import config

def evaluate_system(results_df: pd.DataFrame, test_df: pd.DataFrame, confusion_file: str = "confusion_matrix.png") -> None:
    print("\n--- Final Model Evaluation ---")
    
    # Aggregate Predictions (Logical OR across all stages)
    pred_cols = [f"{stage}_Prediction" for stage in config.STAGE_SENSORS.keys() if f"{stage}_Prediction" in results_df.columns]
    
    # Create a boolean mask: True if ANY stage predicted an attack
    system_alert_mask = (results_df[pred_cols] == config.ATTACK_LABEL).any(axis=1)
    
    # Convert mask back to labels
    results_df['System_Prediction'] = np.where(
        system_alert_mask, 
        config.ATTACK_LABEL, 
        config.NORMAL_LABEL
    )
    
    y_true = test_df[config.LABEL_COL]
    y_pred = results_df['System_Prediction']
    
    # Print Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    
    # Calculate and Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[config.NORMAL_LABEL, config.ATTACK_LABEL])
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['Normal', 'Attack'],
        yticklabels=['Normal', 'Attack']
    )
    plt.title("System-Wide Confusion Matrix")
    plt.ylabel('True State')
    plt.xlabel('Predicted State')
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / confusion_file, dpi=150)
    print(f"\nConfusion matrix saved to {config.OUTPUT_DIR / confusion_file}")
    
    # Final F1 Score Printout
    f1 = f1_score(y_true, y_pred, pos_label=config.ATTACK_LABEL)
    print(f"\nFinal System F1 Score (Attack Class): {f1:.4f}")