import pandas as pd
from sklearn.impute import KNNImputer

def impute_knn(df, numeric_columns, n_neighbors=5):
    """
    paramètres :
    - numeric_columns : liste des colonnes numériques à imputer
    - n_neighbors : nombre de voisins ( 5)

    retour :
    - df_imputed : dataFrame avec les valeurs imputées
    """
    # copier pour ne pas modifier l’original
    df_imputed = df.copy()
    # imputation
    imputer = KNNImputer(n_neighbors=n_neighbors)
    df_imputed[numeric_columns] = imputer.fit_transform(df_imputed[numeric_columns])

    return df_imputed
