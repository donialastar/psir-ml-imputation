"""
train_pipeline.py
─────────────────
Appelle les modules d'imputation, d'entraînement et d’évaluation 
pour entraîner un pipeline complet et l’enregistrer.
"""

from pathlib import Path
import joblib
import pandas as pd

# Appels aux fonctions personnalisées
from imputation.knn import knn_impute
from models.random_forest import train_rf

# ────────────────────────────────────────────────
# 1. Paramètres du dataset et préparation
# ────────────────────────────────────────────────
DATASET_NAME = "hta_nhanes_complet_variables_fr"
RAW_DATA_PATH = Path("C:/Users/bamoi/OneDrive - Groupe ESAIP/$ PSIR/experimentation/repo/psir-ml-imputation/data/raw/data farouk/merged/variables lisibles") / f"{DATASET_NAME}.csv"
PROCESSED_PATH = Path(f"data/processed/{DATASET_NAME}")
MODEL_PATH = Path("src/models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# Sauvegarde brute si pas encore splittée
df = pd.read_csv(RAW_DATA_PATH)
if not (Path(f"data/splits/{DATASET_NAME}/{DATASET_NAME}_train.csv").exists()):
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=0.2, stratify=df["HTA"], random_state=42)

    split_dir = Path(f"data/splits/{DATASET_NAME}")
    split_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(split_dir / f"{DATASET_NAME}_train.csv", index=False)
    test.to_csv(split_dir / f"{DATASET_NAME}_test.csv", index=False)

# ────────────────────────────────────────────────
# 2. Imputation (KNN)
# ────────────────────────────────────────────────
print("Imputation KNN...")
knn_impute(DATASET_NAME, PROCESSED_PATH)

# ────────────────────────────────────────────────
# 3. Entraînement du modèle Random Forest
# ────────────────────────────────────────────────
print("Entraînement du modèle Random Forest...")
metrics_df = train_rf(DATASET_NAME, imputation_method="knn")

# ────────────────────────────────────────────────
# 4. Export du modèle entraîné final
# ────────────────────────────────────────────────
metrics_df, model = train_rf(DATASET_NAME, imputation_method="knn")

# Enregistrement
joblib.dump(model, MODEL_PATH / "pipeline_htn.joblib")
print(f"Modèle final enregistré dans {MODEL_PATH / 'pipeline_htn.joblib'}")
