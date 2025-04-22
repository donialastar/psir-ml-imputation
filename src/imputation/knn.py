import pandas as pd
from sklearn.impute import KNNImputer
from pathlib import Path

def knn_impute(dataset_name, n_neighbors=5):
    train = pd.read_csv(f"data/splits/{dataset_name}/{dataset_name}_train.csv")
    test = pd.read_csv(f"data/splits/{dataset_name}/{dataset_name}_test.csv")

    imputer = KNNImputer(n_neighbors=n_neighbors)
    X_train = pd.DataFrame(imputer.fit_transform(train.drop(columns=['target'])), columns=train.columns[:-1])
    X_test = pd.DataFrame(imputer.transform(test.drop(columns=['target'])), columns=test.columns[:-1])

    output_dir = f"data/processed/{dataset_name}/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pd.concat([X_train, train['target']], axis=1).to_csv(f"{output_dir}/{dataset_name}_knn_train.csv", index=False)
    pd.concat([X_test, test['target']], axis=1).to_csv(f"{output_dir}/{dataset_name}_knn_test.csv", index=False)
