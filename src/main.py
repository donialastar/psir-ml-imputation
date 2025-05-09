"""
main.py
───────
Pipeline centralisé pour tester toutes les combinaisons :
(imputation + modèle), évaluer les performances et sauvegarder
les résultats + modèles + visualisations.

Chaque module (imputation, modèle, métriques, visualisation)
est isolé et ne fait qu’une seule tâche.

Structure prévue :
- splitter.py : génère les splits train/test
- knn.py / mice.py / median.py : génèrent les fichiers imputés
- random_forest.py / dnn.py / grandient_boost.py : entraînent les modèles
- metrics.py : calcule les scores
- visualizations.py : génère les graphiques comparatifs
"""

import os
import pandas as pd
from pathlib import Path
import joblib

# Chargement des modules (chaque module ne fait qu’une chose)
from utils.splitter import split_data
from imputation.knn import knn_impute
from imputation.mice import mice_impute
from imputation.median import median_impute

from models.random_forest import train_rf
from models.dnn import train_dnn
from models.grandient_boost import train_gb

from utils.metrics import calculate_metrics
from utils.visualizations import generate_all_plots

# ───────────────────────────────────────────────
# 1. CONFIGURATION GLOBALE
# ───────────────────────────────────────────────

DATASET_NAME = "hta_nhanes_complet_variables_fr"

# Combinaisons à tester
imputation_methods = ["knn", "mice", "median"]
model_functions = {
    "random_forest": train_rf,
    "dnn": train_dnn,
    "gradient_boost": train_gb
}

# Répertoires de travail
DATA_SPLIT_DIR = Path(f"data/splits/{DATASET_NAME}")
PROCESSED_DIR = Path(f"data/processed/{DATASET_NAME}")
RESULTS_DIR = Path("results/tables") / DATASET_NAME
MODELS_DIR = Path("models") / DATASET_NAME

# Création des dossiers au besoin
for d in [RESULTS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Récupérateur de toutes les métriques
all_results = []

# ───────────────────────────────────────────────
# 2. SPLIT DU DATASET BRUT SI NON FAIT
# ───────────────────────────────────────────────

if not (DATA_SPLIT_DIR / f"{DATASET_NAME}_train.csv").exists():
    print(f"📎 Split du dataset {DATASET_NAME} ...")
    split_dataset(DATASET_NAME)
else:
    print(f"✅ Splits déjà présents pour {DATASET_NAME}")



# ───────────────────────────────────────────────
# 3. TRAITEMENT DE CHAQUE COMBINAISON
# ───────────────────────────────────────────────

for imputation_method in imputation_methods:
    print(f"\n🧩 Étape d’imputation : {imputation_method}")

    # ──────────────
    # Imputation
    # ──────────────
    if imputation_method == "knn":
        knn_impute(DATASET_NAME)
    elif imputation_method == "mice":
        mice_impute(DATASET_NAME)
    elif imputation_method == "median":
        median_impute(DATASET_NAME)
    else:
        print(f"❌ Méthode d’imputation inconnue : {imputation_method}")
        continue

    # ──────────────
    # Chargement des données imputées
    # ──────────────
    train_path = PROCESSED_DIR / f"{DATASET_NAME}_{imputation_method}_imputed_train.csv"
    test_path = PROCESSED_DIR / f"{DATASET_NAME}_{imputation_method}_imputed_test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["HTA"])
    y_train = train_df["HTA"]
    X_test = test_df.drop(columns=["HTA"])
    y_test = test_df["HTA"]

    for model_name, model_func in model_functions.items():
        print(f"\n🚀 Entraînement du modèle : {model_name.upper()} avec {imputation_method.upper()}")

        # ──────────────
        # Entraînement du modèle
        # ──────────────
        model = model_func(X_train, y_train)

        # ──────────────
        # Prédictions
        # ──────────────
        y_pred = model.predict(X_test)
        y_proba = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
            except:
                pass  # certains modèles n'ont pas .predict_proba()

        # ──────────────
        # Évaluation
        # ──────────────
        result_df = calculate_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            nom_modele=model_name,
            nom_imputation=imputation_method
        )

        # Ajout au tableau final
        all_results.append(result_df)

        # ──────────────
        # Sauvegarde du modèle
        # ──────────────
        model_out_dir = MODELS_DIR / f"{model_name}_{imputation_method}"
        model_out_dir.mkdir(parents=True, exist_ok=True)

        model_file = model_out_dir / "pipeline.joblib"
        joblib.dump(model, model_file)

        print(f"💾 Modèle enregistré : {model_file}")


# ───────────────────────────────────────────────
# 4. COMPILATION DES MÉTRIQUES
# ───────────────────────────────────────────────

print("\n📊 Compilation des résultats...")
final_results_df = pd.concat(all_results, ignore_index=True)

# Sauvegarde en CSV
metrics_path = RESULTS_DIR / f"{DATASET_NAME}_all_metrics.csv"
final_results_df.to_csv(metrics_path, index=False)
print(f"📄 Résultats enregistrés dans : {metrics_path}")


# ───────────────────────────────────────────────
# 5. GÉNÉRATION DES VISUALISATIONS
# ───────────────────────────────────────────────

try:
    print("📈 Génération des visualisations...")
    generate_all_plots(final_results_df, output_dir=RESULTS_DIR)
    print("✅ Visualisations sauvegardées dans :", RESULTS_DIR)
except Exception as e:
    print("❌ Échec lors de la génération des visualisations :", e)


# ───────────────────────────────────────────────
# 6. RÉCAPITULATIF DU PIPELINE
# ───────────────────────────────────────────────

print("\n🧾 Résumé du pipeline exécuté :")
print(f"  ➤ Dataset analysé      : {DATASET_NAME}")
print(f"  ➤ Méthodes d’imputation: {', '.join(imputation_methods)}")
print(f"  ➤ Modèles testés       : {', '.join(model_functions.keys())}")
print(f"  ➤ Métriques disponibles : {metrics_path.name}")
print(f"  ➤ Modèles sauvegardés   : {MODELS_DIR}")
print(f"  ➤ Visualisations        : {RESULTS_DIR}")

print("\n🎉 Pipeline terminé avec succès !\n")
