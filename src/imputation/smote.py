import pandas as pd
from imblearn.over_sampling import SMOTE
from pathlib import Path

def apply_smote(dataset_name, input_suffix='knn'):
    train = pd.read_csv(f"data/processed/{dataset_name}/{dataset_name}_{input_suffix}_train.csv")
    X = train.drop(columns=['target'])
    y = train['target']

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    output_dir = f"data/processed/{dataset_name}/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='target')], axis=1) \
        .to_csv(f"{output_dir}/{dataset_name}_{input_suffix}_smote_train.csv", index=False)
