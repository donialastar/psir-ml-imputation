"""
predict.py
──────────
Effectue une prédiction sur un ou plusieurs patients à partir d’un modèle enregistré.

Fonctionnalités :
- Supporte les modèles RandomForest, GradientBoost, DNN
- Chargement depuis models/{dataset}/{model}_{imputation}/pipeline.joblib
- Accepte des fichiers CSV ou une ligne de commande (--features)
- S’aligne dynamiquement sur les colonnes d’origine (avec valeur 0 ou moyenne)
- Retourne une prédiction + degré de confiance
"""

import argparse
import pandas as pd
import joblib
from pathlib import Path
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--imputation', required=True)
    parser.add_argument('--input', help="Fichier CSV avec patients")
    parser.add_argument('--target', default=None)
    parser.add_argument('--features', nargs='*', help="Format : age=45 tension=120")

    return parser.parse_args()

def load_model(dataset, model_name, imputation):
    """
    Charge le modèle depuis le chemin models/... en fonction de la combinaison choisie.
    """
    model_path = Path(f"models/{dataset}/{model_name}_{imputation}/pipeline.joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    model = joblib.load(model_path)
    if isinstance(model, tuple):  # DNN
        return model[0], model[1]
    else:
        return model, None

def load_patient_data(args):
    """
    Charge un patient via fichier CSV ou ligne de commande.
    Supprime la colonne cible si elle est fournie.
    """
    if args.input:
        df = pd.read_csv(args.input)
    elif args.features:
        data_dict = {}
        for pair in args.features:
            key, val = pair.split("=")
            try:
                data_dict[key.strip()] = float(val)
            except:
                data_dict[key.strip()] = val
        df = pd.DataFrame([data_dict])
    else:
        raise ValueError("Aucun patient fourni (via --input ou --features)")

    if args.target and args.target in df.columns:
        df = df.drop(columns=[args.target])

    return df

def align_features(df_input, expected_columns):
    """
    Aligne le DataFrame d’entrée avec les colonnes attendues par le modèle :
    - les colonnes manquantes sont remplies par 0
    - les colonnes inutilisées sont ignorées
    """
    df_aligned = pd.DataFrame(columns=expected_columns)

    for col in expected_columns:
        if col in df_input.columns:
            df_aligned[col] = df_input[col]
        else:
            df_aligned[col] = 0  # Valeur par défaut (ou moyenne si connu)

    return df_aligned

def predict(model, scaler, df_input):
    """
    Applique le modèle à des données partiellement complètes.
    Gère le scaler (si DNN) et renvoie un degré de confiance.
    """
    df_input = df_input.copy()
    df_input.columns = df_input.columns.str.strip().str.lower()

    if scaler:  # Cas DNN
        expected_columns = scaler.feature_names_in_
        df_aligned = align_features(df_input, expected_columns)
        X = scaler.transform(df_aligned)
    else:
        expected_columns = model.feature_names_in_
        df_aligned = align_features(df_input, expected_columns)
        X = df_aligned.values

    # Calcul de la confiance (part des colonnes renseignées)
    n_provided = (df_input.columns.isin(expected_columns)).sum()
    confidence = n_provided / len(expected_columns)

    # Prédiction
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        y_proba = None

    y_pred = model.predict(X)
    return y_pred, y_proba, confidence

def main():
    args = parse_args()

    # Chargement
    model, scaler = load_model(args.dataset, args.model, args.imputation)
    patient_df = load_patient_data(args)

    print(f"Nombre de patients chargés : {len(patient_df)}")

    # Prédiction
    y_pred, y_proba, confidence = predict(model, scaler, patient_df)

    # Affichage
    for i, row in patient_df.iterrows():
        print(f"\n--- Patient {i + 1} ---")
        print(row.to_dict())
        print("Prédiction :", "Hypertendu" if y_pred[i] else "Non hypertendu")
        if y_proba is not None:
            print(f"Probabilité : {y_proba[i]:.3f}")
        print(f"Confiance (colonnes renseignées) : {confidence:.1%}")

if __name__ == "__main__":
    main()
