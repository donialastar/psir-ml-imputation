import pandas as pd
from sklearn.impute import KNNImputer

def knn_impute_data(X):
    """
    Impute les valeurs manquantes dans les données numériques en utilisant KNN.
    
    Args:
        X (DataFrame): Features contenant des valeurs manquantes.
    
    Returns:
        ndarray: Données imputées.
    """
    knn_imputer = KNNImputer(n_neighbors=5)
    X_imputed = knn_imputer.fit_transform(X.select_dtypes(include=['float64']))
    return X_imputed