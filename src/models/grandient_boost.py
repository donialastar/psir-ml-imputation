"""
gradient_boost.py
──────────────────
Entraîne un modèle Gradient Boosting à partir de X_train / y_train.

➡️ Ne lit pas de fichiers.
➡️ Ne calcule pas de métriques.
➡️ Ne sauvegarde rien.
➡️ Appelé par main.py.
"""

from sklearn.ensemble import GradientBoostingClassifier

def train_gb(X_train, y_train):
    """
    Entraîne un Gradient Boosting Classifier.

    Paramètres :
        X_train : DataFrame des variables explicatives
        y_train : Series ou array des labels

    Retourne :
        Un modèle GradientBoostingClassifier entraîné
    """
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
