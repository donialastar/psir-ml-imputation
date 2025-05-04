from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import numpy as np

def get_metrics(y_true, y_pred, y_proba):
    """
    Calcule les métriques de classification : Accuracy, Recall, Specificity, F1-score, AUC
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)

    return {
        "Accuracy": round(accuracy, 4),
        "Recall": round(recall, 4),
        "Specificity": round(specificity, 4),
        "F1-score": round(f1, 4),
        "AUC-ROC": round(auc, 4)
    }

def train_evaluate_rf(X_train, y_train, X_test, y_test, random_state=42):
    """
    Entraîne un RandomForestClassifier et retourne un dictionnaire de métriques
    """
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]  # Probabilité pour la classe 1 (HTA=True)

    # Obtenir les métriques
    scores = get_metrics(y_test, y_pred, y_proba)

    return scores
