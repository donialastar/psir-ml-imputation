import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from pathlib import Path

def mice_impute(input_path, output_path, max_iter=10, random_state=42):
    """
    Imputation MICE avec sauvegarde des résultats
    
    Args:
        input_path: Chemin vers le fichier CSV brut
        output_path: Chemin de sauvegarde
        max_iter: Nombre d'itérations (défaut=10)
        random_state: Reproductibilité
    """
    df = pd.read_csv(input_path)
    
    imputer = IterativeImputer(
        max_iter=max_iter,
        random_state=random_state,
        sample_posterior=True
    )
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_imputed.to_csv(output_path, index=False)
    print(f" Données imputées (MICE) sauvegardées dans {output_path}")