import pandas as pd
from pathlib import Path
from src.utils.splitter import split_data
from src.imputation.knn import knn_impute
from src.imputation.mice import mice_impute
from src.imputation.median import median_impute
from src.imputation.smote import apply_smote
from src.models.dnn import train_dnn
from src.models.random_forest import train_rf
from src.utils.metrics import calculate_metrics
from sklearn.metrics import accuracy_score

# Dictionnaire associant un nom simplifié à un fichier de données
DATASETS = {
    "1": ("ADMISSIONS", "ADMISSIONS.csv"),
    "2": ("heart_disease", "heart_disease.csv"),
    "3": ("nhanes", "Farouk_hta_nhanes_complet_variables_fr.csv"),
    "4": ("hypertension", "Hypertension-risk-model-main.csv")
}

# Méthodes d'imputation disponibles
IMPUTATIONS = ["knn", "mice", "median"]
# Modèles de classification disponibles
MODELS = ["dnn", "rf"]

def run():
    # --- Sélection de l'utilisateur pour le dataset, l'imputation et le modèle ---
    print("\n=== Sélection du dataset ===")
    for k, v in DATASETS.items():
        print(f"{k}. {v[0]} ({v[1]})")
    dataset_key = input("Choisir le dataset (1-4) : ")
    dataset_name, dataset_file = DATASETS.get(dataset_key, (None, None))
    if not dataset_name:
        print("Choix invalide.")
        return

    method = input(f"Choisir la méthode d'imputation {IMPUTATIONS} : ")
    if method not in IMPUTATIONS:
        print("Méthode invalide.")
        return

    model = input(f"Choisir le modèle de classification {MODELS} : ")
    if model not in MODELS:
        print("Modèle invalide.")
        return

    # --- Split si le fichier n'existe pas encore ---
    split_dir = Path(f"data/splits/{dataset_name}/")
    if not (split_dir / f"{dataset_name}_train.csv").exists():
        print("\n--> Split des données...")
        split_data(f"data/raw/{dataset_file}", split_dir, target_col="target", prefix=dataset_name)

    # --- Imputation des données manquantes ---
    print(f"\n--> Imputation avec {method}")
    if method == "knn":
        knn_impute(dataset_name)
    elif method == "mice":
        mice_impute(dataset_name)
    elif method == "median":
        median_impute(dataset_name)

    # --- Application de SMOTE pour équilibrer les classes sur les données d'entraînement ---
    print("--> Application de SMOTE")
    apply_smote(dataset_name=dataset_name, input_suffix=f"{method}_imputed")

    # --- Chargement des données de test imputées ---
    test_df = pd.read_csv(f"data/processed/{dataset_name}/{dataset_name}_{method}_imputed_test.csv")
    y_test = test_df["target"]  # étiquettes vraies
    X_test = test_df.drop(columns=["target"])  # features sans la colonne cible

    # --- Entraînement du modèle + Prédiction sur le test ---
    print(f"\n--> Entraînement avec {model.upper()}")
    if model == "dnn":
        trained_model = train_dnn(dataset_name=dataset_name, imputation_method=method)
        y_pred = (trained_model.predict(X_test) > 0.5).astype(int).flatten()  # seuil à 0.5 pour classer 0 ou 1
    elif model == "rf":
        trained_model = train_rf(dataset_name=dataset_name, imputation_method=method)
        y_pred = trained_model.predict(X_test)

    # --- Calcul des métriques de performance (accuracy, recall, etc.) ---
    metrics = calculate_metrics(y_test, y_pred)

    # --- Sauvegarde des métriques dans le dossier results ---
    results_dir = Path(f"data/results/{dataset_name}/")
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(results_dir / f"{dataset_name}{model}{method}_metrics.csv")

    print("\n Pipeline terminé avec succès !")

if _name_ == "_main_":
    run()

