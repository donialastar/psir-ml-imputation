"""
knn.py
──────
Imputation KNN appliquée aux colonnes numériques.
Les colonnes catégorielles sont remplies par la valeur la plus fréquente (mode).

➡️ Ce module ne fait que transformer les données manquantes et sauvegarder les fichiers imputés.
"""

import pandas as pd
from sklearn.impute import KNNImputer
from pathlib import Path

def knn_impute(dataset_name, target_column="HTA"):
    """
    Imputation KNN pour les colonnes numériques sauf la cible.
    """
    target_column = target_column.strip().lower()

    train_path = f"data/splits/{dataset_name}/{dataset_name}_train.csv"
    test_path = f"data/splits/{dataset_name}/{dataset_name}_test.csv"
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Standardiser noms
    train.columns = train.columns.str.strip().str.lower()
    test.columns = test.columns.str.strip().str.lower()

    categorical_cols = train.select_dtypes(include=['object', 'category']).columns
    numerical_cols = [col for col in train.select_dtypes(include='number').columns if col != target_column]

    train_copy = train.copy()
    test_copy = test.copy()

    imputer = KNNImputer(n_neighbors=5)
    train_copy[numerical_cols] = imputer.fit_transform(train_copy[numerical_cols])
    test_copy[numerical_cols] = imputer.transform(test_copy[numerical_cols])

    for col in categorical_cols:
        mode_val = train_copy[col].mode()[0]
        train_copy[col].fillna(mode_val, inplace=True)
        test_copy[col].fillna(mode_val, inplace=True)

    output_dir = Path(f"data/processed/{dataset_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_copy.to_csv(output_dir / f"{dataset_name}_knn_imputed_train.csv", index=False)
    test_copy.to_csv(output_dir / f"{dataset_name}_knn_imputed_test.csv", index=False)
