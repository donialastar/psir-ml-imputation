import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from ..utils.metrics import calculate_metrics, save_metrics
from pathlib import Path

def train_rf(dataset_name, imputation_method='knn'):
    """
    Entraîne un modèle Random Forest à partir des données imputées + SMOTE.

    Args:
        dataset_name: nom du dataset (ex: "gad")
        imputation_method: méthode utilisée pour l'imputation ('knn', 'mice'...)
    """
    # Chargement des données
    train = pd.read_csv(f"data/processed/{dataset_name}/{dataset_name}_{imputation_method}_smote_train.csv")
    test = pd.read_csv(f"data/processed/{dataset_name}/{dataset_name}_{imputation_method}_test.csv")

    X_train, y_train = train.drop(columns=['target']), train['target']
    X_test, y_test = test.drop(columns=['target']), test['target']

    # Modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Évaluation
    metrics = calculate_metrics(y_test, y_pred)

    # Sauvegarde des résultats
    output_dir = f"results/tables/{dataset_name}/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_metrics(metrics, f"{output_dir}/{dataset_name}_rf_{imputation_method}_metrics.csv")
