from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd

def impute_knn(df, path_out, k=5):
    """
    Applique une imputation KNN sur les colonnes numériques du DataFrame.
    
    Args:
        df (pd.DataFrame): le dataset original avec NaN
        path_out (str): chemin pour enregistrer le fichier imputé (csv)
        k (int): nombre de voisins (par défaut 5)
    """
    df_copy = df.copy()
    
    # Séparer la cible
    y = df_copy["HTA"]
    X = df_copy.drop(columns=["HTA"])

    # Garde uniquement les colonnes numériques
    X_num = X.select_dtypes(include=["float64", "int64"])

    # Standardisation avant KNN
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_num), columns=X_num.columns)

    # Imputation KNN
    imputer = KNNImputer(n_neighbors=k)
    X_imputed = pd.DataFrame(imputer.fit_transform(X_scaled), columns=X_num.columns)

    # Remise à l’échelle originale
    X_unscaled = pd.DataFrame(scaler.inverse_transform(X_imputed), columns=X_num.columns)

    # Remise en forme finale
    df_result = pd.concat([X_unscaled, y.reset_index(drop=True)], axis=1)
    df_result.to_csv(path_out, index=False)
    print(f"✅ Dataset imputé par KNN sauvegardé : {path_out}")
