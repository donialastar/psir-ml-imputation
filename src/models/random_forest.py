import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from ..utils.metrics import calculate_metrics

def train_rf(data_path, target_col, n_estimators=100, cv=5):
    """
    Entraînement Random Forest avec validation croisée
    
    Args:
        data_path: Chemin vers les données imputées
        target_col: Nom de la colonne cible
        n_estimators: Nombre d'arbres (défaut=100)
        cv: Nombre de folds (défaut=5)
    """
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        class_weight='balanced'
    )
    
    y_pred = cross_val_predict(model, X, y, cv=cv)
    return calculate_metrics(y, y_pred)