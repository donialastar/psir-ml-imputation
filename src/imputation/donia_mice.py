import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

def impute_mice(df, numeric_columns, max_iter=10, random_state=42):
    """
     paramètres :
    - numeric_columns : liste des colonnes numériques à imputer
    - max_iter : nombre de cycles d’imputation

    retour :
    - df_imputed : dataFrame avec les valeurs imputées
    """
    # copier pour ne pas modifier le dataset de base
    df_imputed = df.copy()
    # imputation
    imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
    # applique l’imputation uniquement sur les colonnes numeric_coloumns
    df_imputed[numeric_columns] = imputer.fit_transform(df_imputed[numeric_columns])

    return df_imputed
