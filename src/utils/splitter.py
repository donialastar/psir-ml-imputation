"""
splitter.py
───────────
Lit un dataset brut, nettoie la colonne cible, et le divise en train/test stratifié.

Fonctionnalités :
- Gestion des cibles booléennes, textuelles, numériques ou continues
- Conversion des colonnes cibles en binaire si besoin
- Nettoyage des NaN
- Standardisation des noms de colonnes
- Sauvegarde des splits dans data/splits/{dataset}/
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def split_dataset(dataset_name, target_column="HTA", test_size=0.2, random_state=42):
    """
    Splitte un dataset en train/test stratifié.

    Paramètres :
        dataset_name : str
            Nom du fichier CSV (sans extension), présent dans data/raw/
        target_column : str
            Nom de la colonne cible à prédire
        test_size : float
            Proportion du test set
        random_state : int
            Graine pour reproductibilité
    """

    # Charger le fichier CSV depuis le dossier brut
    raw_path = Path(f"C:/Users/bamoi/OneDrive - Groupe ESAIP/$ PSIR/experimentation/repo/psir-ml-imputation/data/raw/{dataset_name}.csv")
    if not raw_path.exists():
        raise FileNotFoundError(f"Dataset introuvable : {raw_path}")

    df = pd.read_csv(raw_path)

    # Standardiser les noms de colonnes
    df.columns = df.columns.str.strip().str.lower()
    target_column = target_column.strip().lower()

    # Vérifier que la colonne cible existe
    if target_column not in df.columns:
        raise ValueError(f"Colonne cible '{target_column}' absente du dataset.")
    
    # Remplacer les chaînes 'nan' littérales par des vrais NaN
    df[target_column] = df[target_column].replace('nan', pd.NA)

    # Si c’est une colonne texte, convertir proprement
    if df[target_column].dtype == 'object' or df[target_column].dtype == 'string':
        df[target_column] = df[target_column].astype(str).str.strip().str.lower().map({
            'yes': 1, 'no': 0,
            'true': 1, 'false': 0,
            '1': 1, '0': 0
        })

    elif pd.api.types.is_numeric_dtype(df[target_column]):
        unique_vals = df[target_column].dropna().unique()
        if len(unique_vals) > 2:
            # Si la colonne contient des mesures continues (ex: pression)
            print(f"Colonne '{target_column}' détectée comme continue, conversion binaire avec seuil ≥140")
            df[target_column] = df[target_column].apply(lambda x: 1 if x >= 140 else 0)

    # Si après mapping c’est toujours du texte ou avec NaN, convertir le reste manuellement
    if df[target_column].isna().any():
        # Essayer conversion directe en entier
        try:
            df[target_column] = df[target_column].astype(float)
        except:
            print("Valeurs non convertibles :", df[target_column].unique())
            raise ValueError(f"Certaines valeurs de la colonne cible '{target_column}' sont invalides ou manquantes après conversion.")

    # Supprimer les lignes avec valeur cible manquante
    initial_len = len(df)
    df = df.dropna(subset=[target_column])
    if len(df) < initial_len:
        print(f"{initial_len - len(df)} ligne(s) supprimée(s) car la cible '{target_column}' était NaN")

    # Séparer les données
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Split stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Fusionner à nouveau pour export
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Sauvegarde des splits
    split_dir = Path(f"data/splits/{dataset_name}")
    split_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(split_dir / f"{dataset_name}_train.csv", index=False)
    test_df.to_csv(split_dir / f"{dataset_name}_test.csv", index=False)

    print(f"Split terminé pour {dataset_name} : {len(train_df)} train / {len(test_df)} test")
