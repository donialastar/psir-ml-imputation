import pandas as pd
from sklearn.impute import KNNImputer
from pathlib import Path

def knn_impute(dataset_name):
    """
    Impute les données train/test existantes et sauvegarde dans processed/
    Args:
        dataset_name: 'donia', 'gad', etc. (doit correspondre à votre structure)
    """
    # Chemins d'entrée (déjà splités)
    train_path = f"data/splits/{dataset_name}/{dataset_name}_train.csv"
    test_path = f"data/splits/{dataset_name}/{dataset_name}_test.csv"
    
    # Chargement
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Séparation X/y
    X_train, y_train = train.drop(columns=['target']), train['target']
    X_test, y_test = test.drop(columns=['target']), test['target']
    
    # Imputation (fit sur train seulement)
    imputer = KNNImputer(n_neighbors=5)
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)
    
    # Sauvegarde dans processed/
    output_dir = f"data/processed/{dataset_name}/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    pd.concat([X_train_imp, y_train], axis=1).to_csv(f"{output_dir}/{dataset_name}_knn_train.csv", index=False)
    pd.concat([X_test_imp, y_test], axis=1).to_csv(f"{output_dir}/{dataset_name}_knn_test.csv", index=False)