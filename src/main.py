"""
main.py
───────
Pipeline complet multi-datasets pour la prédiction de l'hypertension artérielle.
Fonctionnalités :
- Imputation KNN, MICE, médiane
- Encodage des colonnes catégorielles
- Application de SMOTE sur X_train / y_train uniquement
- Entraînement sur plusieurs modèles
- Évaluation et visualisation
"""

import os
import pandas as pd
import joblib
from pathlib import Path
from imputation.smote import apply_smote
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# ─── Imports locaux ──────────────────────────────
from utils.splitter import split_dataset
from imputation.knn import knn_impute
from imputation.mice import mice_impute
from imputation.median import median_impute
from models.random_forest import train_rf
from models.dnn import train_dnn
#from models.grandient_boost import train_gb
from utils.metrics import calculate_metrics
from utils.visualizations import generate_all_plots

# ─── Configuration des datasets ──────────────────
datasets = [
    {
        "file": "Farouk_hta_nhanes_complet_variables_fr",
        "target": "HTA"
    },
    {
        "file": "Hypertension-risk-model-main",
        "target": "Risk"
    },
    {
        "file": "heart_disease",
        "target": "High Blood Pressure"
    },
    #{
    #    "file": "ADMISSIONS",
    #    "target": "HTA"
    #}
]

imputation_methods = ["knn", "mice", "median"]

model_functions = {
    "random_forest": train_rf,
    "dnn": train_dnn,
    #"gradient_boost": train_gb
}

global_results = []

scenario_counter = 1

# ─── Pipeline principal par dataset ───────────────
for dataset_config in datasets:
    DATASET_NAME = dataset_config["file"]
    TARGET_COLUMN = dataset_config["target"]
    target_col = TARGET_COLUMN.strip().lower()

    print(f"\nTRAITEMENT : {DATASET_NAME} (target = '{target_col}')")

    DATA_SPLIT_DIR = Path(f"data/splits/{DATASET_NAME}")
    PROCESSED_DIR = Path(f"data/processed/{DATASET_NAME}")
    RESULTS_DIR = Path("results/tables") / DATASET_NAME
    MODELS_DIR = Path("models") / DATASET_NAME

    for d in [RESULTS_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # ─── Étape 1 : split du dataset si nécessaire ───
    if not (DATA_SPLIT_DIR / f"{DATASET_NAME}_train.csv").exists():
        split_dataset(DATASET_NAME, target_column=target_col)
    else:
        print(f"Split déjà effectué pour {DATASET_NAME}")


    dataset_results = []

    # ─── Étape 2 : boucle sur les méthodes d’imputation ───
    for imputation_method in imputation_methods:
        print(f"\nImputation : {imputation_method.upper()}")

        if imputation_method == "knn":
            knn_impute(DATASET_NAME, target_column=target_col)
        elif imputation_method == "mice":
            mice_impute(DATASET_NAME, target_column=target_col)
        elif imputation_method == "median":
            median_impute(DATASET_NAME, target_column=target_col)

        # Chargement des fichiers imputés
        train_df = pd.read_csv(PROCESSED_DIR / f"{DATASET_NAME}_{imputation_method}_imputed_train.csv")
        test_df = pd.read_csv(PROCESSED_DIR / f"{DATASET_NAME}_{imputation_method}_imputed_test.csv")

        # Standardisation des noms de colonnes
        train_df.columns = train_df.columns.str.strip().str.lower()
        test_df.columns = test_df.columns.str.strip().str.lower()

        if target_col not in train_df.columns:
            raise ValueError(f"Colonne cible '{target_col}' introuvable dans {train_df.columns.tolist()}")

        # Séparation features / labels
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]

        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]

        # Vérification de duplication entre train et test
        duplicate_rows = pd.merge(X_train, X_test, how='inner')
        if len(duplicate_rows) > 0:
            print(f"{len(duplicate_rows)} lignes en commun entre train et test. Risque de sur-apprentissage.")
        else:
            print("Train/Test bien séparés")

        # Affichage des distributions
        print("Distribution de la cible :")
        print("Train :", y_train.value_counts(normalize=True).to_dict())
        print("Test  :", y_test.value_counts(normalize=True).to_dict())

        # ─── Étape 3 : encodage des colonnes catégorielles ───
        # Inclure les booléens avec les numériques (0/1)
        numerical_cols = X_train.select_dtypes(include=['number', 'bool']).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

        if categorical_cols:
            print(f"Colonnes catégorielles détectées : {categorical_cols}")

            preprocessor = ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ], remainder='passthrough')

            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)

            # Conversion en DataFrame pour compatibilité modèles
            X_train = pd.DataFrame(X_train.toarray() if hasattr(X_train, "toarray") else X_train)
            X_test = pd.DataFrame(X_test.toarray() if hasattr(X_test, "toarray") else X_test)
        else:
            print("Aucun encodage requis.")

        # Conversion explicite en float64 (meilleur support KNN/DNN)
        X_train = X_train.astype("float64")
        X_test = X_test.astype("float64")

        # ─── Étape 4 : remplacement des cibles textuelles (juste au cas où) ───
        if y_train.dtype == "object":
            y_train = y_train.str.strip().str.lower().map({'yes': 1, 'no': 0, 'true': 1, 'false': 0})
        if y_test.dtype == "object":
            y_test = y_test.str.strip().str.lower().map({'yes': 1, 'no': 0, 'true': 1, 'false': 0})

        # ─── Étape 5 : application de SMOTE ───
        if y_train.nunique() == 2:
            print("Application de SMOTE (équilibrage classes)...")
            X_train, y_train = apply_smote(X_train, y_train)
            print("Distribution après SMOTE :", y_train.value_counts().to_dict())
        else:
            print("SMOTE non appliqué (la cible n’est pas binaire)")

        
        # ─── Étape 6 : boucle sur les modèles à entraîner ───
        for model_name, model_func in model_functions.items():
            print(f"\nEntraînement : {model_name.upper()} sur {imputation_method.upper()}")

            if model_name == "dnn":
                # Le modèle DNN nécessite un scaler (StandardScaler appliqué dans dnn.py)
                model, scaler = model_func(X_train, y_train)
                X_test_scaled = scaler.transform(X_test)
                y_pred = (model.predict(X_test_scaled) > 0.5).astype(int).flatten()
                y_proba = None  # Optionnel : implémenter predict_proba personnalisé
            else:
                model = model_func(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            # ─── Évaluation ───
            result_df = calculate_metrics(
                y_true=y_test,
                y_pred=y_pred,
                y_proba=y_proba,
                nom_modele=model_name,
                nom_imputation=imputation_method
            )
            
            result_df["Scenario ID"] = f"S{scenario_counter}"
            result_df["Dataset"] = DATASET_NAME
            scenario_counter += 1

            # Réorganiser les colonnes pour que Scenario ID et Dataset soient devant
            cols = result_df.columns.tolist()
            if "Scenario ID" in cols and "Dataset" in cols:
                cols = ["Scenario ID", "Dataset"] + [col for col in cols if col not in ["Scenario ID", "Dataset"]]
                result_df = result_df[cols]
            
            dataset_results.append(result_df)

            # ─── Sauvegarde du modèle ───
            model_dir = MODELS_DIR / f"{model_name}_{imputation_method}"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Suppression du modèle précédent
            model_path = model_dir / "pipeline.joblib"
            if model_path.exists():
                model_path.unlink()

            if model_name == "dnn":
                joblib.dump((model, scaler), model_dir / "pipeline.joblib")
            else:
                joblib.dump(model, model_dir / "pipeline.joblib")

            print(f"Modèle enregistré : {model_dir / 'pipeline.joblib'}")

    # ─── Étape 7 : compilation des résultats pour le dataset ───
    if dataset_results:
        final_df = pd.concat(dataset_results, ignore_index=True)

        # Supprimer les doublons par Dataset + Modèle + Imputation
        final_df = final_df.drop_duplicates(subset=["Dataset", "Modèle", "Imputation"])

        global_results.append(final_df)

        # Sauvegarde propre par dataset
        metrics_path = RESULTS_DIR / f"{DATASET_NAME}_all_metrics.csv"
        final_df.to_csv(metrics_path, index=False)
        print(f"Résultats enregistrés : {metrics_path}")

        # Visualisations Plotly
        try:
            generate_all_plots(final_df, output_dir=RESULTS_DIR)
            print(f"Graphiques générés dans : {RESULTS_DIR}")
        except Exception as e:
            print(f"Erreur de visualisation : {e}")

# ───────────────────────────────────────────────
# 6. SYNTHÈSE GLOBALE MULTI-DATASETS (facultatif)
# ───────────────────────────────────────────────

if global_results:
    print("\nFusion des résultats multi-datasets...")
    report_df = pd.concat(global_results, ignore_index=True)
    global_csv_path = Path("results/all_datasets_metrics.csv")
    global_csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(global_csv_path, index=False)
    print(f"Résumé global enregistré : {global_csv_path}")
    
    # === GÉNÉRATION DES 16 GRAPHIQUES FINAUX ===
    summary_df = pd.read_csv("results/all_datasets_metrics.csv")
    summary_df.columns = summary_df.columns.str.strip().str.lower()

    if "nom_dataset" in summary_df.columns:
        summary_df.rename(columns={"nom_dataset": "dataset"}, inplace=True)
    if "modèle" in summary_df.columns:
        summary_df.rename(columns={"modèle": "modèle"}, inplace=True)
    if "imputation method" in summary_df.columns:
        summary_df.rename(columns={"imputation method": "imputation"}, inplace=True)

    from utils.visualizations import plot_metric_by_dataset
    metrics_to_plot = ["accuracy", "recall", "precision", "f1-score"]
    for metric in metrics_to_plot:
        plot_metric_by_dataset(summary_df, metric, output_dir="results/summary_plots")

print("\nPipeline terminé avec succès pour tous les datasets !")


