"""
predict_patient.py
──────────────────
Recharge le pipeline entraîné (models/pipeline_htn.joblib)
et prédit HTA + probabilité pour UN nouveau patient passé
en ligne de commande.

Exemple :
    python src/predict_patient.py \
        --age 52 --imc 28.4 --sys 140 --chol 205
"""

import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ───────────────────────────────────────────────
# 1. Argument parser : valeurs du nouveau patient
# ───────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Prédiction HTA pour un patient")
parser.add_argument("--age",   type=float, required=True, help="Âge (années)")
parser.add_argument("--imc",   type=float, required=True, help="Indice de Masse Corporelle")
parser.add_argument("--sys",   type=float, required=True, help="TA systolique moyenne (mmHg)")
parser.add_argument("--chol",  type=float, required=True, help="Cholestérol total (mg/dL)")
args = parser.parse_args()

# ───────────────────────────────────────────────
# 2. Charger le pipeline entraîné
# ───────────────────────────────────────────────
model_path = Path("C:/Users/bamoi/OneDrive - Groupe ESAIP/$ PSIR/experimentation/repo/psir-ml-imputation/src/models/pipeline_htn.joblib")
if not model_path.exists():
    raise FileNotFoundError("❌ models/pipeline_htn.joblib introuvable. "
                            "Lance d’abord train_pipeline.py")

pipe = joblib.load(model_path)

# ───────────────────────────────────────────────
# 3. Construire le DataFrame du patient
# ───────────────────────────────────────────────
patient_dict = {
    "age_annees": args.age,
    "imc": args.imc,
    "tension_sys_moy_mmHg": args.sys,
    "cholesterol_total_mgdl": args.chol
    # ➜ si ton modèle utilise d’autres colonnes
    #    ajoute-les ici (NaN si inconnues)
}
df_patient = pd.DataFrame([patient_dict])

# ↳ ajoute les colonnes manquantes (NaN)
for col in pipe.feature_names_in_:
    if col not in df_patient.columns:
        df_patient[col] = np.nan

# 3. ré-ordonne les colonnes exactement comme à l’entraînement
df_patient = df_patient[pipe.feature_names_in_]

# Tout convertir en float (obligatoire pour KNNImputer)
df_patient = df_patient.astype("float64")


# ───────────────────────────────────────────────
# 4. Prédictions
# ───────────────────────────────────────────────
classe = pipe.predict(df_patient)[0]          # 0 ou 1
proba  = pipe.predict_proba(df_patient)[0, 1] # proba d’HTA

# ───────────────────────────────────────────────
# 5. Affichage résultat
# ───────────────────────────────────────────────
print("\n─────────  Résultat  ─────────")
print("Hypertension :", "Oui" if classe else "Non")
print("Probabilité  :", f"{proba:.3f}")
print("────────────────────────────────")
