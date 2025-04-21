import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def train_model(patient_data_train, diagnostic_train, input_dim=None, epochs=50, batch_size=32):
    """
    paramètres :
    - input_dim : nombre de colonnes (auto-détecté si on ne le précise pas
    - epochs : nombre de fois que le modele va passer sur tout le data set (50 fois)
    - batch_size : nombre d'exemplaires à traiter à la fois (32)

    retour :
    - modèle entraîné
    """
    if input_dim is None:
        input_dim = patient_data_train.shape[1]  # nombre de colonnes (features)

    # initialise un modele vide
    model = Sequential()
    # 1er couche
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    # 2e couche
    model.add(Dense(32, activation='relu'))
    # 1 seul neurone ( malade ou pas )
    model.add(Dense(1, activation='sigmoid'))  # donne un score entre 0 et 1

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(patient_data_train, diagnostic_train, epochs=epochs, batch_size=batch_size, verbose=1)

    return model


def predict(model, patient_data_test):
    """
    prédit le diagnostic avec le modèle DNN entraîné.

    paramètres :
    - model : modèle Keras entraîné
    - patient_data_test : données d'entrée à prédire

    Retour :
    - y_pred : prédictions binaires (0 ou 1)
    """
    raw_preds = model.predict(patient_data_test)
    y_pred = (raw_preds > 0.5).astype(int)  # seuil de 0.5 pour classer en 0 ou 1
    return y_pred
