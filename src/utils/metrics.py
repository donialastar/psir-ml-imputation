"""
metrics.py
──────────
Calcule une large gamme de métriques de classification binaire :
- accuracy, recall, specificity, precision, F1, F2
- ROC AUC, PR AUC
- balanced accuracy, MCC, log loss
- confusion matrix (tn, fp, fn, tp)
Retourne un DataFrame avec tous les résultats.

➡️ Ne sauvegarde pas les métriques (à faire dans main.py).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    balanced_accuracy_score,
    matthews_corrcoef,
    log_loss
)

def calculate_metrics(
    y_true,
    y_pred,
    nom_modele,
    nom_imputation,
    y_proba=None
):
    """
    Calcule une large gamme de métriques pour la classification binaire.

    Paramètres :
        y_true        : array-like, Ground truth (0 ou 1)
        y_pred        : array-like, Prédictions binaires
        nom_modele    : str, nom du modèle (ex: 'Random Forest')
        nom_imputation: str, méthode d’imputation utilisée
        y_proba       : array-like, Probabilités associées à la classe positive (optionnel)

    Retourne :
        DataFrame à une ligne contenant toutes les métriques
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Scores classiques
    accuracy   = accuracy_score(y_true, y_pred)
    precision  = precision_score(y_true, y_pred, zero_division=0)
    recall     = recall_score(y_true, y_pred, zero_division=0)
    f1         = f1_score(y_true, y_pred, zero_division=0)
    f2         = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Scores avec proba (si fournie)
    auc_roc    = roc_auc_score(y_true, y_proba) if y_proba is not None else None
    auc_pr     = average_precision_score(y_true, y_proba) if y_proba is not None else None
    logloss    = log_loss(y_true, y_proba) if y_proba is not None else None

    # Autres
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    mcc          = matthews_corrcoef(y_true, y_pred)

    # Compilation dans un DataFrame
    result = {
        "Imputation": nom_imputation,
        "Modèle": nom_modele,
        "Accuracy": round(accuracy, 4),
        "Balanced_Accuracy": round(balanced_acc, 4),
        "Recall": round(recall, 4),
        "Specificity": round(specificity, 4),
        "Precision": round(precision, 4),
        "F1-score": round(f1, 4),
        "F2-score": round(f2, 4),
        "MCC": round(mcc, 4),
        "AUC-ROC": round(auc_roc, 4) if auc_roc is not None else None,
        "AUC-PR": round(auc_pr, 4) if auc_pr is not None else None,
        "LogLoss": round(logloss, 4) if logloss is not None else None,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn
    }

    return pd.DataFrame([result])
