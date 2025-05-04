import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def split_data(input_path, output_dir, target_col, test_size=0.2, prefix="farouk"):
    """
    Divise les données 80/20 et sauvegarde avec le format [prefix]_train.csv et [prefix]_test.csv
    
    Args:
        input_path: Chemin vers le fichier CSV d'origine
        output_dir: Dossier de sortie (créé automatiquement si inexistant)
        target_col: Nom de la colonne cible
        prefix: Préfixe pour les noms de fichiers (ex: "gad")
    """
    # Chargement des données
    data = pd.read_csv(input_path)
    
    # Split stratifié
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop(columns=[target_col]),
        data[target_col],
        test_size=test_size,
        stratify=data[target_col],
        random_state=42
    )
    
    # Création du dossier si besoin
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Sauvegarde avec les noms spécifiés
    pd.concat([X_train, y_train], axis=1).to_csv(f"{output_dir}/{prefix}_train.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(f"{output_dir}/{prefix}_test.csv", index=False)
    
    print(f" Données splitées sauvegardées dans {output_dir}/")
    print(f"  - {prefix}_train.csv : {len(X_train)} échantillons")
    print(f"  - {prefix}_test.csv : {len(X_test)} échantillons")