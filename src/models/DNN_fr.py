from sklearn.neural_network import MLPClassifier
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

def train_evaluate_mlp(X_train, y_train, X_test, y_test, random_state=42):
    """
    Entraîne un perceptron multicouche (MLP) et retourne un dictionnaire de métriques.
    """
    clf = MLPClassifier(
        hidden_layer_sizes=(100, 50),  # 2 couches cachées avec 100 et 50 neurones
        activation='relu',             # Fonction d’activation classique
        solver='adam',                 # Optimiseur robuste
        max_iter=300,                  # Nombre d’itérations
        random_state=random_state
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]  # Probabilité pour HTA = 1

    # Calcul des métriques
    scores = get_metrics(y_test, y_pred, y_proba)

    return scores
