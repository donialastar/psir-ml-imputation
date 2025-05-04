import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from ..utils.metrics import calculate_metrics, save_metrics
from pathlib import Path

def train_dnn(dataset_name, imputation_method='knn'):
    train = pd.read_csv(f"data/processed/{dataset_name}/{dataset_name}_{imputation_method}_smote_train.csv")
    test = pd.read_csv(f"data/processed/{dataset_name}/{dataset_name}_{imputation_method}_test.csv")

    X_train, y_train = train.drop(columns=['target']), train['target']
    X_test, y_test = test.drop(columns=['target']), test['target']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

    y_pred = (model.predict(X_test_scaled) > 0.5).astype(int).flatten()
    metrics = calculate_metrics(y_test, y_pred)
    save_metrics(metrics, f"results/tables/{dataset_name}/{dataset_name}_dnn_{imputation_method}_metrics.csv")
