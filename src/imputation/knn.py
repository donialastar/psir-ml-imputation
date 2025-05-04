from sklearn.impute import KNNImputer
import pandas as pd
from pathlib import Path

def knn_impute(dataset_name, output_dir):
    """
    Applique l'imputation KNN en différenciant les colonnes numériques et catégorielles.
    Args:
        dataset_name: Nom du dataset (ex: 'gad')
    """
    # Chargement des splits existants
    train_path = f"data/splits/{dataset_name}/{dataset_name}_train.csv"
    test_path = f"data/splits/{dataset_name}/{dataset_name}_test.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Séparation des colonnes numériques et catégorielles
    categorical_cols = train.select_dtypes(include=['object']).columns
    numerical_cols = train.select_dtypes(exclude=['object']).columns

    # Copie des données pour ne pas altérer les données de base
    train_copy = train.copy()
    test_copy = test.copy()
    
    # Imputation KNN pour les données numériques
    imputer = KNNImputer(n_neighbors=5)
    train_copy[numerical_cols] = imputer.fit_transform(train_copy[numerical_cols])
    test_copy[numerical_cols] = imputer.transform(test_copy[numerical_cols])
    
    # Imputation pour les données catégorielles par la mode
    for col in categorical_cols:
        train_copy[col].fillna(train_copy[col].mode()[0], inplace=True)
        test_copy[col].fillna(test_copy[col].mode()[0], inplace=True)

    # Sauvegarde après imputation
    output_dir = f"data/processed/{dataset_name}/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    train_copy.to_csv(f"{output_dir}/{dataset_name}_knn_imputed_train.csv", index=False)
    test_copy.to_csv(f"{output_dir}/{dataset_name}_knn_imputed_test.csv", index=False)
