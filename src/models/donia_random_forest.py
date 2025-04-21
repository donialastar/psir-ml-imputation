import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_model(patient_data_train, diagnostic_train):
    """
    paramètres :
    - patient_data_train : les caractéristiques des patients (ex : âge, tension...)
    - diagnostic_train : colonnes des diagnostics réels (
    retour :
    - modèle entraîné
    """

    # on crée un modèle Random Forest avec 100 arbres
    model = RandomForestClassifier(n_estimators=100, random_state=42)

    # On entraîne le modèle avec les données d'entrée et les diagnostics connus
    model.fit(patient_data_train, diagnostic_train)

    # On retourne le modèle entraîné
    return model

# Fonction qui utilise le modèle entraîné pour faire des prédictions
def predict(model, patient_data_test):
    """
    prédit le diagnostic à partir des données de nouveaux patients.

    paramètres :
    - model : modèle Random Forest déjà entraîné
    - patient_data_test : DataFrame contenant les caractéristiques des patients à prédire

    retour :
    - y_pred : tableau des prédictions (diagnostics prévus par le modèle)
    """
    
    y_pred = model.predict(patient_data_test)
    return y_pred


