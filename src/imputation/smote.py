from imblearn.over_sampling import SMOTE
import pandas as pd
from pathlib import Path

def apply_smote(dataset_name, input_suffix='knn'):
    """
    Applique SMOTE sur les données d'entraînement avant imputation.
    Args:
        dataset_name: Nom du dataset (ex: 'gad')
        input_suffix: Suffixe pour déterminer si on applique sur KNN ou MICE (ex: 'knn')
    """
    # Chargement des données d'entraînement (avant l'imputation)
    train_path = f"data/splits/{dataset_name}/{dataset_name}_train.csv"
    test_path = f"data/splits/{dataset_name}/{dataset_name}_test.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Séparer les features et la target
    X_train, y_train = train.drop(columns=['target']), train['target']
    X_test, y_test = test.drop(columns=['target']), test['target']
    
    # Appliquer SMOTE sur l'entraînement pour équilibrer les classes
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Sauvegarde des données après SMOTE
    output_dir = f"data/processed/{dataset_name}/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    pd.concat([X_train_smote, y_train_smote], axis=1).to_csv(
        f"{output_dir}/{dataset_name}_{input_suffix}_smote_train.csv", index=False
    )
    pd.concat([X_test, y_test], axis=1).to_csv(
        f"{output_dir}/{dataset_name}_{input_suffix}_smote_test.csv", index=False
    )
