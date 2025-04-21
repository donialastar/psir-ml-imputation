import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from ..utils.metrics import calculate_metrics

def train_dnn(data_path, target_col, epochs=50, batch_size=32):
    """
    Entraînement DNN avec prétraitement automatique
    
    Args:
        data_path: Chemin vers les données imputées
        target_col: Nom de la colonne cible
        epochs: Nombre d'epochs (défaut=50)
        batch_size: Taille des batches (défaut=32)
    """
    df = pd.read_csv(data_path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Architecture
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Entraînement avec validation split
    model.fit(
        X_scaled, y,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0
    )
    
    y_pred = (model.predict(X_scaled) > 0.5).astype(int).flatten()
    return calculate_metrics(y, y_pred)