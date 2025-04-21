import pandas as pd
from sklearn.impute import KNNImputer
from pathlib import Path

def knn_impute(input_path, output_path, n_neighbors=5, cols_with_missing=None):
    """
    Imputation KNN avec sauvegarde des résultats
    
    Args:
        input_path: Chemin vers le fichier CSV brut
        output_path: Chemin de sauvegarde
        n_neighbors: Nombre de voisins (défaut=5)
        cols_with_missing: Liste des colonnes à imputer (si None, impute toutes les NaN)
    """
    df = pd.read_csv(input_path)
    
    if cols_with_missing:
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df[cols_with_missing] = imputer.fit_transform(df[cols_with_missing])
    else:
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f" Données imputées (KNN) sauvegardées dans {output_path}")