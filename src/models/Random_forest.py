

from sklearn.ensemble import RandomForestClassifier

def create_random_forest_model():
    """
    Crée un modèle de classification Random Forest avec des paramètres par défaut.
    
    Returns:
        RandomForestClassifier: Un modèle prêt à être entraîné avec .fit(X, y)
    """

    model = RandomForestClassifier(
        n_estimators=100,      # Nombre d'arbres dans la forêt
        max_depth=None,        # Laisse l'arbre se développer jusqu'à ce que toutes les feuilles soient pures
        random_state=42        # Graine aléatoire pour résultats reproductibles
    )
    return model

