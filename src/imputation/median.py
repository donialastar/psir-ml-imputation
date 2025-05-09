import pandas as pd
from pathlib import Path

def median_impute(dataset_name):
    """
    Applique l'imputation par médiane (colonnes numériques) et mode (colonnes catégorielles).
    Sauvegarde les données imputées dans le dossier processed.

    Args:
        dataset_name: Nom du dataset (ex: 'donia', 'gad')
    """
    # 1. Chargement des données
    train_path = f"data/splits/{dataset_name}/{dataset_name}_train.csv"
    test_path = f"data/splits/{dataset_name}/{dataset_name}_test.csv"
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # 2. Séparation des types de colonnes
    numerical_cols = train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = train.select_dtypes(include=['object', 'category']).columns

    # 3. Copie des données pour imputation
    train_imp = train.copy()
    test_imp = test.copy()

    # 4. Imputation numérique (médiane)
    for col in numerical_cols:
        median_val = train[col].median()
        train_imp[col].fillna(median_val, inplace=True)
        test_imp[col].fillna(median_val, inplace=True)

    # 5. Imputation catégorielle (mode)
    for col in categorical_cols:
        mode_val = train[col].mode()[0]  # Prend le premier mode si plusieurs
        train_imp[col].fillna(mode_val, inplace=True)
        test_imp[col].fillna(mode_val, inplace=True)

    # 6. Sauvegarde
    output_dir = Path(f"data/processed/{dataset_name}/")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_imp.to_csv(output_dir / f"{dataset_name}_median_imputed_train.csv", index=False)
    test_imp.to_csv(output_dir / f"{dataset_name}_median_imputed_test.csv", index=False)

    print(f"Imputation par médiane/mode terminée pour {dataset_name}")
    print(f"Fichiers sauvegardés dans {output_dir}/")