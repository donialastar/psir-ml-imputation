"""
median.py
─────────
Applique une imputation par la médiane pour les colonnes numériques,
et la valeur la plus fréquente (mode) pour les colonnes catégorielles.

➡️ Ce module lit uniquement les fichiers de split
et sauvegarde les fichiers imputés dans `data/processed/`.
"""

import pandas as pd
from pathlib import Path

def median_impute(dataset_name, target_column="HTA"):
    target_column = target_column.strip().lower()

    train_path = f"data/splits/{dataset_name}/{dataset_name}_train.csv"
    test_path = f"data/splits/{dataset_name}/{dataset_name}_test.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train.columns = train.columns.str.strip().str.lower()
    test.columns = test.columns.str.strip().str.lower()

    categorical_cols = train.select_dtypes(include=['object', 'category']).columns
    numerical_cols = [col for col in train.select_dtypes(include='number').columns if col != target_column]

    train_copy = train.copy()
    test_copy = test.copy()

    for col in numerical_cols:
        median_val = train_copy[col].median()
        train_copy[col].fillna(median_val, inplace=True)
        test_copy[col].fillna(median_val, inplace=True)

    for col in categorical_cols:
        mode_val = train_copy[col].mode()[0]
        train_copy[col].fillna(mode_val, inplace=True)
        test_copy[col].fillna(mode_val, inplace=True)

    output_dir = Path(f"data/processed/{dataset_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_copy.to_csv(output_dir / f"{dataset_name}_median_imputed_train.csv", index=False)
    test_copy.to_csv(output_dir / f"{dataset_name}_median_imputed_test.csv", index=False)