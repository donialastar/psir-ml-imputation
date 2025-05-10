"""
smote.py
────────
Applique SMOTE (Synthetic Minority Over-sampling Technique)
pour équilibrer les classes dans un jeu de données.

Utilisé après imputation, avant entraînement.
Appliqué uniquement si la variable cible est binaire.
"""

from imblearn.over_sampling import SMOTE
import pandas as pd

def apply_smote(X, y, random_state=42):
    """
    Applique SMOTE uniquement si la cible est binaire.

    Paramètres :
        X : DataFrame ou array
            Features (déjà imputées et encodées)
        y : Series ou array
            Cible binaire
        random_state : int
            Graine aléatoire (par défaut 42)

    Retourne :
        (X_resampled, y_resampled)
    """
    # Vérifie que la cible est bien binaire
    if isinstance(y, pd.Series):
        unique_vals = y.dropna().unique()
    else:
        y = pd.Series(y)
        unique_vals = y.dropna().unique()

    if len(unique_vals) != 2:
        print("SMOTE ignoré : la cible n'est pas binaire.")
        return X, y

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    #Protection : reconvertir X en DataFrame si c'était un DataFrame
    if isinstance(X, pd.DataFrame):
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)

    return X_resampled, y_resampled
