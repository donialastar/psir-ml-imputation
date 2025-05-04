"""
predict_batch.py
─────────────────
Recharge le pipeline entraîné (`models/pipeline_htn.joblib`)
et prédit HTA (classe + probabilité) pour **tout un CSV** de nouveaux patients.

Exemple d’appel :
    python src/predict_batch.py \
        --in  data/nouveaux_patients.csv \
        --out results/preds_htn.csv
"""

import argparse
import joblib
import pandas as pd
from pathlib import Path

# ───────────────────────────────────────────────
# 1. Arguments CLI
# ───────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Prédire HTA sur un fichier CSV")
parser.add_argument("--in",  dest="infile",  required=True,
                    help="Chemin du CSV à prédire")
parser.add_argument("--out", dest="outfile", required=True,
                    help="Chemin du CSV de sortie (prédictions)")
args = parser.parse_args()

in_path  = Path(args.infile)
out_path = Path(args.outfile)

if not in_path.exists():
    raise FileNotFoundError(f"❌ Fichier d'entrée introuvable : {in_path}")

# ───────────────────────────────────────────────
# 2. Charger le pipeline
# ───────────────────────────────────────────────
model_path = Path("models/pipeline_htn.joblib")
if not model_path.exists():
    raise FileNotFoundError("❌ models/pipeline_htn.joblib manquant : "
                            "exécute d'abord train_pipeline.py")
pipe = joblib.load(model_path)

# ───────────────────────────────────────────────
# 3. Charger le CSV des patients
# ───────────────────────────────────────────────
df_in = pd.read_csv(in_path)

# Vérifie que toutes les colonnes nécessaires sont présentes
missing_cols = [c for c in pipe.feature_names_in_ if c not in df_in.columns]
if missing_cols:
    # ajoute les colonnes manquantes remplies de NaN
    df_in[missing_cols] = pd.NA
    print(f"⚠️  Colonnes manquantes ajoutées (NaN) : {missing_cols}")

# Réordonne les colonnes comme dans l'entraînement
df_in = df_in[pipe.feature_names_in_]

# ───────────────────────────────────────────────
# 4. Prédictions
# ───────────────────────────────────────────────
classes = pipe.predict(df_in)              # 0 / 1
probas  = pipe.predict_proba(df_in)[:, 1]  # proba classe 1

df_out = df_in.copy()
df_out["_HTA_pred"]  = classes
df_out["_HTA_proba"] = probas.round(3)

# ───────────────────────────────────────────────
# 5. Sauvegarde
# ───────────────────────────────────────────────
out_path.parent.mkdir(parents=True, exist_ok=True)
df_out.to_csv(out_path, index=False)
print(f"✅ Prédictions sauvegardées : {out_path}")
