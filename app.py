import streamlit as st
import pandas as pd
import numpy as np
import pickle
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

# Функции предобработки как в Colab
def clean_numeric(df):
    df = df.copy()
    for col in ['mileage', 'engine', 'max_power']:
        if col in df.columns:
            df[col] = (df[col]
                      .astype(str)
                      .str.replace(r'[^\d\.]+', '', regex=True)
                      .replace('', np.nan)
                      .astype(float))
    return df

def preprocess_input(df):
    df = df.copy()
    df = df.drop(columns=['torque'], errors='ignore')
    df = clean_numeric(df)
    df['brand'] = df['name'].str.split().str[0]
    df = df.drop(columns=['name'], errors='ignore')
    return df

# Загружаем модель и данные
try:
    model = load_model()
    data = load_preprocessed_data()
    
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction from CSV"])
    
    with tab1:
        # Загружаем сырые данные для справочных значений
        @st.cache_data
        def load_raw_data():
            url = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv"
            return pd.read_csv(url)
        
        df_raw = load_raw_data()
        
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
        
        if st.sidebar.button("Predict Price", type="primary"):
            # Создаём входные данные в формате исходного датасета
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
            
            # Преобразуем в DataFrame (1 строка)
            input_df = pd.DataFrame([input_dict])
            
            # Предобрабатываем
            processed_df = preprocess_input(input_df)
            
            # Убедимся что все колонки есть
            for col in data['feature_names']:
                if col not in processed_df.columns:
                    processed_df[col] = 0
            
            # Упорядочиваем колонки как в обучении
            processed_df = processed_df[data['feature_names']]
            
            # Предсказываем
            prediction = model.predict(processed_df)[0]
            
            st.success(f"### Predicted Price: ₹{prediction:,.2f}")
            
            # Показываем детали
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Raw Input")
                st.dataframe(input_df, use_container_width=True)
            
            with col2:
                st.subheader("Processed Features")
                st.dataframe(processed_df, use_container_width=True)
    
    with tab2:
        st.header("Batch Prediction from CSV")
        st.write("Upload a CSV file with the same format as the original dataset")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Читаем CSV
                batch_df = pd.read_csv(uploaded_file)
                
                # Проверяем необходимые колонки
                required_cols = ['name', 'year', 'km_driven', 'fuel', 'seller_type', 
                                'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
                
                missing_cols = [col for col in required_cols if col not in batch_df.columns]
                
                if missing_cols:
                    st.error(f"Missing columns in CSV: {missing_cols}")
                else:
                    st.success(f"CSV loaded successfully: {len(batch_df)} rows")
                    
                    # Показываем предпросмотр
                    st.subheader("CSV Preview")
                    st.dataframe(batch_df.head(), use_container_width=True)
                    
                    if st.button("Process and Predict Batch", type="primary"):
                        with st.spinner("Processing..."):
                            # Сохраняем оригинальные данные
                            original_df = batch_df.copy()
                            
                            # Предобработка
                            processed_batch = preprocess_input(batch_df)
                            
                            # Добавляем недостающие колонки
                            for col in data['feature_names']:
                                if col not in processed_batch.columns:
                                    processed_batch[col] = 0
                            
                            # Упорядочиваем
                            processed_batch = processed_batch[data['feature_names']]
                            
                            # Предсказываем
                            predictions = model.predict(processed_batch)
                            
                            # Добавляем предсказания к оригинальному DataFrame
                            original_df['predicted_price'] = predictions
                            
                            # Показываем результаты
                            st.subheader("Predictions")
                            st.dataframe(original_df[['name', 'year', 'fuel', 'engine', 'predicted_price']], 
                                       use_container_width=True)
                            
                            # Статистика
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Cars", len(original_df))
                            with col2:
                                st.metric("Avg Price", f"₹{original_df['predicted_price'].mean():,.0f}")
                            with col3:
                                st.metric("Max Price", f"₹{original_df['predicted_price'].max():,.0f}")
                            
                            # Скачать результаты
                            csv = original_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="Download Predictions CSV",
                                data=csv,
                                file_name="car_price_predictions.csv",
                                mime="text/csv"
                            )
                            
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")
        
        # Пример формата CSV
        with st.expander("Show CSV Format Example"):
            example_data = {
                'name': ['Maruti Suzuki Swift', 'Hyundai i20', 'Honda City'],
                'year': [2018, 2020, 2019],
                'km_driven': [45000, 25000, 35000],
                'fuel': ['Petrol', 'Diesel', 'Petrol'],
                'seller_type': ['Individual', 'Dealer', 'Individual'],
                'transmission': ['Manual', 'Automatic', 'Manual'],
                'owner': ['First Owner', 'First Owner', 'Second Owner'],
                'mileage': [20.4, 18.0, 17.5],
                'engine': [1197, 1493, 1498],
                'max_power': [82.0, 100.0, 115.0],
                'seats': [5, 5, 5],
                'torque': ['113Nm@4400rpm', '240Nm@1500rpm', '145Nm@4600rpm']
            }
            example_df = pd.DataFrame(example_data)
            st.dataframe(example_df, use_container_width=True)
            
            # Кнопка для скачивания примера
            csv_example = example_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Example CSV",
                data=csv_example,
                file_name="example_cars.csv",
                mime="text/csv"
            )

except FileNotFoundError:
    st.error("Model files not found. Please upload:")
    st.write("1. `simple_model.pkl` - trained model")
    st.write("2. `preprocessed_data.pkl` - preprocessed training data")
    
    # Альтернатива: тренировать модель на лету
    if st.button("Train Model Now (takes time)"):
        with st.spinner("Training model..."):
            # Код для обучения модели на лету
            pass

except Exception as e:
    st.error(f"Error: {str(e)[:200]}")
