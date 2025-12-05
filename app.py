import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import io

st.set_page_config(page_title="Car Price Prediction", layout="wide")
st.title('Car Price Prediction')

@st.cache_resource
def load_model():
    with open("simple_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_preprocessed_data():
    with open("preprocessed_data.pkl", "rb") as f:
        return pickle.load(f)

# Функция предобработки как в Colab
def preprocess_input(input_dict, data):
    # Создаём DataFrame с тем же форматом
    df = pd.DataFrame([input_dict])
    
    # Предобработка
    df = df.drop(columns=['torque'], errors='ignore')
    
    # Чистим числовые поля
    for col in ['mileage', 'engine', 'max_power']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Добавляем бренд
    df['brand'] = df['name'].str.split().str[0]
    df = df.drop(columns=['name'])
    
    # Убедимся что все колонки есть
    for col in data['feature_names']:
        if col not in df.columns:
            df[col] = 0
    
    return df[data['feature_names']]

# Загружаем
try:
    model = load_model()
    data = load_preprocessed_data()
    df_raw = pd.read_csv("https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv")
    
    st.sidebar.header("Input Car Details")
    
    year = st.sidebar.number_input("Year", 1980, 2024, 2015)
    km_driven = st.sidebar.number_input("KM Driven", 0, 1000000, 50000)
    mileage = st.sidebar.number_input("Mileage", 0.0, 50.0, 20.0)
    engine = st.sidebar.number_input("Engine (CC)", 0, 5000, 1200)
    max_power = st.sidebar.number_input("Max Power", 0.0, 500.0, 80.0)
    
    fuel = st.sidebar.selectbox("Fuel Type", df_raw['fuel'].unique())
    seller_type = st.sidebar.selectbox("Seller Type", df_raw['seller_type'].unique())
    transmission = st.sidebar.selectbox("Transmission", df_raw['transmission'].unique())
    owner = st.sidebar.selectbox("Owner", df_raw['owner'].unique())
    seats = st.sidebar.selectbox("Seats", sorted(df_raw['seats'].dropna().unique().astype(int)))
    
    brand_names = df_raw['name'].str.split().str[0].unique()
    brand = st.sidebar.selectbox("Brand", sorted(brand_names))
    
    if st.sidebar.button("Predict Price"):
        # Создаём входные данные
        input_dict = {
            'name': f"{brand} Model",
            'year': year,
            'km_driven': km_driven,
            'fuel': fuel,
            'seller_type': seller_type,
            'transmission': transmission,
            'owner': owner,
            'mileage': mileage,
            'engine': engine,
            'max_power': max_power,
            'seats': seats,
            'torque': '100Nm@2000rpm'
        }
        
        # Предобрабатываем
        X_input = preprocess_input(input_dict, data)
        
        # Предсказываем
        prediction = model.predict(X_input)[0]
        
        st.success(f"### Predicted Price: ₹{prediction:,.2f}")
        
        # Показываем данные
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input Features")
            st.dataframe(X_input)
        
        with col2:
            st.subheader("Model Info")
            st.write(f"Features: {len(data['feature_names'])}")
            st.write(f"Train samples: {len(data['X_train'])}")
    
    # Информация о данных
    with st.expander("Preprocessed Data Info"):
        st.write(f"X_train shape: {data['X_train'].shape}")
        st.write(f"y_train shape: {data['y_train'].shape}")
        st.write(f"Features: {data['feature_names']}")

except FileNotFoundError as e:
    st.error(f"File not found: {e}")
    st.info("Please upload preprocessed_data.pkl and simple_model.pkl")
    
except Exception as e:
    st.error(f"Error: {str(e)[:200]}")
