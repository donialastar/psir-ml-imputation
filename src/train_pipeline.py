"""
train_pipeline.py
─────────────────
Crée un pipeline complet
(imputation KNN  → scaling  → Random Forest),
l’entraîne sur le dataset HTA
et l’enregistre dans models/pipeline_htn.joblib
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# ────────────────────────────────────────────────
# 1. Chargement  du jeu de données complet
# ────────────────────────────────────────────────
data_path = "C:/Users/bamoi/OneDrive - Groupe ESAIP/$ PSIR/experimentation/repo/psir-ml-imputation/data/raw/data farouk/merged/variables lisibles/hta_nhanes_complet_variables_fr.csv"
df = pd.read_csv(data_path)

# Séparation cible / features
y = df["HTA"]
X = df.drop(columns=["HTA"])

# Garder uniquement les colonnes numériques (le KNNImputer les gère bien)
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
X_num = X[num_cols]

# Train / test split pour contrôle (80 % / 20 %)
X_train, X_test, y_train, y_test = train_test_split(
    X_num, y, test_size=0.2, stratify=y, random_state=42
)

# ────────────────────────────────────────────────
# 2. Construction du pipeline
# ────────────────────────────────────────────────
pipe = Pipeline([
    ("imputer", KNNImputer(n_neighbors=5)),
    ("scaler",  StandardScaler()),
    ("clf",     RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1))
])

# ────────────────────────────────────────────────
# 3. Entraînement
# ────────────────────────────────────────────────
pipe.fit(X_train, y_train)

# (facultatif) évaluation rapide sur le test set
score = pipe.score(X_test, y_test)
print(f"Accuracy rapide sur test : {score:.3f}")

# ────────────────────────────────────────────────
# 4. Sauvegarde du pipeline
# ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent   # racine du dépôt
MODEL_DIR = ROOT / "src/models"
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(pipe, MODEL_DIR / "pipeline_htn.joblib")
print(f"enregistré dans {MODEL_DIR / 'pipeline_htn.joblib'}")
