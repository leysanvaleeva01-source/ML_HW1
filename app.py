import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import requests
import io
import sys
import types
!pip install scikit-learn==1.6.1
st.set_page_config(page_title="Car Price Prediction", layout="wide")
st.title('Car Price Prediction App')

sys.modules['__main__'].clean_numeric = None
sys.modules['__main__'].drop_torque = None
sys.modules['__main__'].add_brand = None

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

@st.cache_resource
def load_pipeline():
    try:
        with open("final_pipeline.pkl", "rb") as f:
            pipeline = pickle.load(f)
        st.success("Pipeline loaded")
        return pipeline
    except Exception as e:
        st.error(f"Error loading pipeline: {e}")
        return None

def load_data():
    try:
        url = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv"
        df = pd.read_csv(url)
        return df
    except:
        url = "https://raw.githubusercontent.com/your-username/your-repo/main/Car%20Details%20v4.csv"
        response = requests.get(url)
        df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        return df

df_raw = load_data()
pipeline = load_pipeline()

if pipeline is not None:
    st.sidebar.header("Input Car Details")
    
    year = st.sidebar.number_input("Year", min_value=1980, max_value=2024, value=2015)
    km_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, value=50000)
    mileage = st.sidebar.number_input("Mileage", min_value=0.0, value=20.0)
    engine = st.sidebar.number_input("Engine (CC)", min_value=0, value=1200)
    max_power = st.sidebar.number_input("Max Power (bhp)", min_value=0.0, value=80.0)
    
    fuel = st.sidebar.selectbox("Fuel Type", df_raw['fuel'].unique())
    seller_type = st.sidebar.selectbox("Seller Type", df_raw['seller_type'].unique())
    transmission = st.sidebar.selectbox("Transmission", df_raw['transmission'].unique())
    owner = st.sidebar.selectbox("Owner", df_raw['owner'].unique())
    seats = st.sidebar.selectbox("Seats", sorted(df_raw['seats'].dropna().unique()))
    
    brand_names = df_raw['name'].str.split().str[0].unique()
    brand = st.sidebar.selectbox("Brand", sorted(brand_names))
    
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
    
    if st.sidebar.button("Predict Price"):
        try:
            prediction = pipeline.predict(input_data)[0]
            st.success(f"Predicted Price: ₹{prediction:,.2f}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
    
    with st.expander("Raw Data"):
        st.dataframe(df_raw.head(20))
        
    df_processed = df_raw.copy()
    df_processed = drop_torque(df_processed)
    df_processed = clean_numeric(df_processed)
    df_processed = add_brand(df_processed)
    
    with st.expander("Processed Data"):
        st.dataframe(df_processed.head(20))
