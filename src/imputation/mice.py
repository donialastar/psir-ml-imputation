import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from pathlib import Path

def mice_impute(dataset_name):
    """
    Impute avec MICE et sauvegarde dans processed/
    Args:
        dataset_name: 'donia', 'gad'... (doit correspondre à vos dossiers splits/)
    """
    # Chargement des splits existants
    train_path = f"data/splits/{dataset_name}/{dataset_name}_train.csv"
    test_path = f"data/splits/{dataset_name}/{dataset_name}_test.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Imputation (fit uniquement sur train)
    imp = IterativeImputer(
        max_iter=10,
        random_state=42,
        initial_strategy='mean'
    )
    X_train = pd.DataFrame(
        imp.fit_transform(train.drop(columns=['target'])),
        columns=train.columns[:-1]
    )
    X_test = pd.DataFrame(
        imp.transform(test.drop(columns=['target'])),
        columns=test.columns[:-1]
    )
    
    # Sauvegarde
    output_dir = f"data/processed/{dataset_name}/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    pd.concat([X_train, train['target']], axis=1).to_csv(
        f"{output_dir}/{dataset_name}_mice_train.csv",
        index=False
    )
    pd.concat([X_test, test['target']], axis=1).to_csv(
        f"{output_dir}/{dataset_name}_mice_test.csv",
        index=False
    )