import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import Ridge

def clean_numeric(X):
    X = X.copy()
    for col in ['mileage', 'engine', 'max_power']:
        X[col] = (X[col]
                  .astype(str)
                  .str.replace(r'[^\d\.]+', '', regex=True)
                  .replace('', np.nan)
                  .astype(float))
    return X

def drop_torque(X):
    return X.drop(columns=['torque'], errors='ignore')

def add_brand(X):
    X = X.copy()
    X['brand'] = X['name'].str.split().str[0]
    X = X.drop(columns=['name'])
    return X

cleaner_numeric = FunctionTransformer(clean_numeric)
drop_torque_tf = FunctionTransformer(drop_torque)
brand_tf = FunctionTransformer(add_brand)

numeric_features = ['year', 'km_driven', 'mileage', 'engine', 'max_power']
categorical_features = ['fuel', 'seller_type', 'seats', 'transmission', 'owner', 'brand']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
])

final_pipeline = Pipeline([
    ('drop_torque', drop_torque_tf),
    ('cleaner_numeric', cleaner_numeric),
    ('brand_creator', brand_tf),
    ('preprocessor', preprocessor),
    ('model', Ridge(alpha=1.0))
])

url = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv"
df = pd.read_csv(url)
X = df.drop('selling_price', axis=1)
y = df['selling_price']

final_pipeline.fit(X, y)

with open("final_pipeline.pkl", "wb") as f:
    pickle.dump(final_pipeline, f, protocol=4)

print("Pipeline saved to final_pipeline.pkl")
print("Sklearn version used:", sklearn.__version__)
