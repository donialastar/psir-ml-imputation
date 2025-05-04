from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

import pandas as pd

def calculate_metrics(y_true, y_pred, nom_modele, nom_imputation, y_proba=None,):
    """
    Calcule un dictionnaire complet de métriques de classification binaire.

    Paramètres :
        y_true : array-like (ground truth)
        y_pred : array-like (prédictions binaires)
        y_proba : array-like (probabilités associées à la classe positive)
        nom_modele : nom du modèle (ex: 'Random Forest')
        nom_imputation : nom de la méthode d’imputation (ex: 'KNN')

    Retourne :
        dict contenant les métriques et les identifiants du test
    """

    # Matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calcul des métriques
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    specificity = float(tn) / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba) if y_proba is not None else None

    metrics_dict = {
        "Imputation": nom_imputation,
        "Modèle": nom_modele,
        "Accuracy": round(float(accuracy), 4),
        "Recall":   round(float(recall),   4),
        "Specificity": round(float(specificity), 4),
        "Precision":   round(float(precision),   4),
        "F1-score":    round(float(f1),          4),
        "AUC-ROC":     round(float(auc), 4) if auc is not None else None
    }
    
    return pd.DataFrame([metrics_dict])