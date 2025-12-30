import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

def impute_mice(df, path_out):
    """
    Applique l'imputation MICE (régression itérative) sur les colonnes numériques.

    Args:
        df (pd.DataFrame): le dataset d'origine avec NaN
        path_out (str): chemin pour sauvegarder le fichier imputé
    """
    df_copy = df.copy()

    # Séparation de la cible
    y = df_copy["HTA"]
    X = df_copy.drop(columns=["HTA"])

    # Sélection des colonnes numériques uniquement
    X_num = X.select_dtypes(include=["float64", "int64"])

    # Standardisation
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_num), columns=X_num.columns)

    # Imputation MICE (régression par défaut = BayesianRidge)
    imputer = IterativeImputer(random_state=0, max_iter=10)
    X_imputed = pd.DataFrame(imputer.fit_transform(X_scaled), columns=X_scaled.columns)

    # Remise à l'échelle d'origine
    X_unscaled = pd.DataFrame(scaler.inverse_transform(X_imputed), columns=X_scaled.columns)

    # Reconstruction finale du dataset imputé
    df_result = pd.concat([X_unscaled, y.reset_index(drop=True)], axis=1)
    df_result.to_csv(path_out, index=False)

    print(f"✅ Dataset imputé par MICE sauvegardé : {path_out}")
