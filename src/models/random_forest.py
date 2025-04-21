from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from ..utils.metrics import calculate_metrics, save_metrics  # Importation des fonctions de métriques
import pandas as pd
from pathlib import Path

def train_rf(dataset_name, imputation_method='knn'):
    """
    Entraîne un modèle Random Forest sur les données imputées
    Args:
        dataset_name: 'donia', 'gad'...
        imputation_method: 'knn' ou 'mice'
    """
    # Chargement des données imputées et SMOTE
    train_path = f"data/processed/{dataset_name}/{dataset_name}_{imputation_method}_smote_train.csv"
    test_path = f"data/processed/{dataset_name}/{dataset_name}_{imputation_method}_test.csv"
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Séparation des features et de la cible
    X_train, y_train = train.drop(columns=['target']), train['target']
    X_test, y_test = test.drop(columns=['target']), test['target']
    
    # Entraînement du modèle Random Forest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Prédictions
    y_pred = model.predict(X_test)
    
    # Calcul des métriques
    metrics = calculate_metrics(y_test, y_pred)
    
    # Sauvegarde des résultats
    output_dir = f"results/tables/{dataset_name}/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Sauvegarde des métriques dans un fichier CSV
    save_metrics(metrics, f"{output_dir}/{dataset_name}_rf_{imputation_method}_metrics.csv")
