import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from utils.metrics import calculate_metrics
from pathlib import Path

def train_rf(dataset_name, imputation_method='knn'):
    """
    Entraîne un modèle Random Forest à partir des données imputées + SMOTE.

    Args:
        dataset_name: nom du dataset (ex: "gad")
        imputation_method: méthode utilisée pour l'imputation ('knn', 'mice'...)
    """
    # Chargement des données
    train = pd.read_csv(f"data/processed/{dataset_name}/{dataset_name}_{imputation_method}_imputed_train.csv")
    test = pd.read_csv(f"data/processed/{dataset_name}/{dataset_name}_{imputation_method}_imputed_test.csv")

    X_train, y_train = train.drop(columns=['HTA']), train['HTA']
    X_test, y_test = test.drop(columns=['HTA']), test['HTA']

    # Modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    y_proba = model.predict_proba(X_test)[:, 1]   # proba d’être dans la classe 1


    # Évaluation
    metrics = calculate_metrics(y_test, y_pred,"random forest", "knn",y_proba)

    # ---------- sauvegarde ----------
    out_dir = Path(f"results/tables/{dataset_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / f"{dataset_name}_rf_{imputation_method}_metrics.csv",
                   index=False)

    # ---------- renvoi ----------
    return metrics            # ← on retourne le DataFrame

