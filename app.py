import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from sklearn.compose import ColumnTransformer
import joblib
import sklearn
print(sklearn.__version__)

st.set_page_config(page_title="Car Price Prediction", layout="wide")
st.title('Car Price Prediction')

@st.cache_resource
def load_pipeline():
    try:
        # Попробуем joblib вместо pickle
        pipeline = joblib.load("final_pipeline.pkl")
        return pipeline
    except:
        # Если не работает, создадим пайплайн заново
        st.warning("Loading from pickle failed, creating new pipeline...")
        
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
            ('ohe', OneHotEncoder(handle_unknown='infrequent_if_exist', sparse_output=False))
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
        
        return final_pipeline

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv"
    return pd.read_csv(url)

pipeline = load_pipeline()
df_raw = load_data()

st.sidebar.header("Input Car Details")

year = st.sidebar.slider("Year", 1980, 2024, 2015)
km_driven = st.sidebar.number_input("Kilometers Driven", 0, 1000000, 50000)
mileage = st.sidebar.number_input("Mileage (km/l or km/kg)", 0.0, 50.0, 20.0)
engine = st.sidebar.number_input("Engine (CC)", 0, 5000, 1200)
max_power = st.sidebar.number_input("Max Power (bhp)", 0.0, 500.0, 80.0)

fuel = st.sidebar.selectbox("Fuel Type", df_raw['fuel'].unique())
seller_type = st.sidebar.selectbox("Seller Type", df_raw['seller_type'].unique())
transmission = st.sidebar.selectbox("Transmission", df_raw['transmission'].unique())
owner = st.sidebar.selectbox("Owner", df_raw['owner'].unique())
seats = st.sidebar.selectbox("Seats", sorted(df_raw['seats'].dropna().unique().astype(int)))

brand_names = df_raw['name'].str.split().str[0].unique()
brand = st.sidebar.selectbox("Brand", sorted(brand_names))

if st.sidebar.button("Predict Price"):
    input_data = pd.DataFrame({
        'name': [f"{brand} Model"],
        'year': [year],
        'km_driven': [km_driven],
        'fuel': [fuel],
        'seller_type': [seller_type],
        'transmission': [transmission],
        'owner': [owner],
        'mileage': [mileage],
        'engine': [engine],
        'max_power': [max_power],
        'seats': [seats],
        'torque': ['100Nm@2000rpm']
    })
    
    try:
        prediction = pipeline.predict(input_data)[0]
        st.success(f"Predicted Price: ₹{prediction:,.2f}")
        
        st.subheader("Input Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Year:** {year}")
            st.write(f"**KM Driven:** {km_driven:,}")
            st.write(f"**Mileage:** {mileage}")
            st.write(f"**Engine:** {engine} CC")
            st.write(f"**Max Power:** {max_power} bhp")
        with col2:
            st.write(f"**Fuel:** {fuel}")
            st.write(f"**Seller Type:** {seller_type}")
            st.write(f"**Transmission:** {transmission}")
            st.write(f"**Owner:** {owner}")
            st.write(f"**Seats:** {seats}")
            st.write(f"**Brand:** {brand}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Training model on the fly...")
        
        X = df_raw.drop('selling_price', axis=1)
        y = df_raw['selling_price']
        pipeline.fit(X, y)
        
        prediction = pipeline.predict(input_data)[0]
        st.success(f"Predicted Price (after training): ₹{prediction:,.2f}")

st.subheader("Dataset Preview")
st.dataframe(df_raw.head(10))
