import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # activation du module
from sklearn.impute import IterativeImputer

# Séparer la cible si besoin (ici 'Risk') pour ne pas l’imputer
target = df2['Risk']
features = df2.drop(columns=['Risk'])

# Imputation MICE avec IterativeImputer
imputer = IterativeImputer(random_state=0)
features_imputed = imputer.fit_transform(features)

# Reconstruction du DataFrame avec les noms d'origine
df_imputed = pd.DataFrame(features_imputed, columns=features.columns)

# Réintégration de la colonne 'Risk'
df_imputed['Risk'] = target
