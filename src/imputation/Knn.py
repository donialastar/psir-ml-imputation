import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
# Créer l'imputer KNN avec 5 voisins
imputer = KNNImputer(n_neighbors=5)

# Appliquer l'imputation sur le dataframe
df1_imputed = pd.DataFrame(imputer.fit_transform(df1), columns=df1.columns)

# Vérification
print(df1_imputed.isnull().sum())
