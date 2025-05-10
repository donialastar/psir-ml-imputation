"""
random_forest.py
────────────────
Contient une fonction pour entraîner un modèle Random Forest 
à partir de X_train et y_train. Ne lit pas de fichier, ne fait 
pas d’évaluation, ne sauvegarde rien.

➡️ Utilisé par main.py
"""

from sklearn.ensemble import RandomForestClassifier

def train_rf(X_train, y_train):
    """
    Entraîne un modèle Random Forest.

    Paramètres :
        X_train : DataFrame des variables explicatives
        y_train : Series ou array des labels

    Retourne :
        Un modèle RandomForestClassifier entraîné
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    return model
