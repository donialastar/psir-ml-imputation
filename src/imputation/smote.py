from imblearn.over_sampling import SMOTE
import pandas as pd
from pathlib import Path

def apply_smote(dataset_name):
    """
    Applique SMOTE uniquement sur le train déjà imputé
    Args:
        dataset_name: Nom du dataset (doit correspondre à vos dossiers)
    """
    # Chargement des données imputées
    input_path = f"data/processed/{dataset_name}/{dataset_name}_knn_train.csv"
    df_train = pd.read_csv(input_path)
    
    # SMOTE (uniquement sur le train)
    X_res, y_res = SMOTE(random_state=42).fit_resample(
        df_train.drop(columns=['target']),
        df_train['target']
    )
    
    # Sauvegarde
    output_path = f"data/processed/{dataset_name}/{dataset_name}_knn_smote_train.csv"
    pd.concat([X_res, y_res], axis=1).to_csv(output_path, index=False)
    
    # Copie du test non modifié (important!)
    test_path = f"data/processed/{dataset_name}/{dataset_name}_knn_test.csv"
    pd.read_csv(test_path).to_csv(
        f"data/processed/{dataset_name}/{dataset_name}_knn_smote_test.csv", 
        index=False
    )