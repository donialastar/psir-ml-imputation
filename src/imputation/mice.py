import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def mice_impute_data(X):
    """
    Impute les valeurs manquantes en utilisant l'algorithme MICE.
    
    Args:
        X (DataFrame): Features contenant des valeurs manquantes.
    
    Returns:
        ndarray: Données imputées.
    """
    mice_imputer = IterativeImputer(random_state=42)
    X_imputed = mice_imputer.fit_transform(X.select_dtypes(include=['float64']))
    return X_imputed