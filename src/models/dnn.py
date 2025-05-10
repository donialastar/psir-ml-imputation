"""
dnn.py
──────
Contient une fonction pour entraîner un modèle Deep Neural Network 
(DNN) à partir de X_train et y_train, avec preprocessing local 
(scaling).

➡️ Ne lit pas les fichiers, ne fait pas d’évaluation, ne sauvegarde rien.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler

def train_dnn(X_train, y_train):
    from tensorflow.keras import backend as K
    K.clear_session()
    """
    Entraîne un réseau de neurones dense.

    Paramètres :
        X_train : DataFrame des variables explicatives
        y_train : Series ou array des labels

    Retourne :
        Un modèle Keras entraîné
    """
    # Normalisation des features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Définition du modèle
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_scaled.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Entraînement
    model.fit(X_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    # On retourne à la fois le modèle et le scaler, utile pour prédictions
    return (model, scaler)
