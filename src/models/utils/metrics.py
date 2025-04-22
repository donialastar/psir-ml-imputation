from sklearn.metrics import accuracy_score, recall_score, f1_score
import pandas as pd

def calculate_metrics(y_true, y_pred, average='binary'):
    return pd.DataFrame({
        'Accuracy': [round(accuracy_score(y_true, y_pred), 4)],
        'Recall': [round(recall_score(y_true, y_pred, average=average, zero_division=0), 4)],
        'F1-score': [round(f1_score(y_true, y_pred, average=average, zero_division=0), 4)]
    })

def save_metrics(metrics_df, output_path):
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metrics_df.to_csv(output_path, index=False)
