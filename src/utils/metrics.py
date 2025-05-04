import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

def calculate_metrics(y_true, y_pred):
    """
    Calcule les métriques de base pour un problème de classification binaire.

    Args:
        y_true: vraies étiquettes
        y_pred: étiquettes prédites

    Returns:
        DataFrame pandas avec : accuracy, precision, recall, f1_score
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    metrics_dict = {
        "accuracy": [acc],
        "precision": [prec],
        "recall": [rec],
        "f1_score": [f1]
    }

    return pd.DataFrame(metrics_dict)
