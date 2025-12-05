import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Предсказание цен на автомобили", layout="wide")
st.title('Предсказание цен на автомобили')

@st.cache_resource
def load_model():
    with open("simple_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_preprocessed_data():
    with open("preprocessed_data.pkl", "rb") as f:
        return pickle.load(f)

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

try:
    model = load_model()
    data = load_preprocessed_data()
    
    tab1, tab2, tab3, tab4= st.tabs(["Предсказание для одного авто", "Пакетное предсказание из CSV", "EDA и Статистика датасета", "Веса модели"])
    
    with tab1:
        st.header("Предсказание цены для одного автомобиля")
        
        col1, col2 = st.columns(2)
        
        with col1:
            year = st.number_input("Год выпуска", 1900, 2025, 2015)
            km_driven = st.number_input("Пробег (км)", 0, 1000000, 50000)
            mileage = st.number_input("Расход топлива", 0.0, 500.0, 20.0)
            engine = st.number_input("Объем двигателя (см³)", 0, 500000, 1200)
            max_power = st.number_input("Мощность (л.с.)", 0.0, 5000.0, 80.0)
        
        with col2:
            fuel_options = data['X_train']['fuel'].unique() if 'fuel' in data['X_train'].columns else []
            seller_options = data['X_train']['seller_type'].unique() if 'seller_type' in data['X_train'].columns else []
            transmission_options = data['X_train']['transmission'].unique() if 'transmission' in data['X_train'].columns else []
            owner_options = data['X_train']['owner'].unique() if 'owner' in data['X_train'].columns else []
            seats_options = sorted(data['X_train']['seats'].dropna().unique().astype(int)) if 'seats' in data['X_train'].columns else []
            brand_options = sorted(data['X_train']['brand'].unique()) if 'brand' in data['X_train'].columns else []
            
            fuel = st.selectbox("Тип топлива", fuel_options) if len(fuel_options) > 0 else st.selectbox("Тип топлива", ["Petrol", "Diesel"])
            seller_type = st.selectbox("Тип продавца", seller_options) if len(seller_options) > 0 else st.selectbox("Тип продавца", ["Individual", "Dealer"])
            transmission = st.selectbox("Трансмиссия", transmission_options) if len(transmission_options) > 0 else st.selectbox("Трансмиссия", ["Manual", "Automatic"])
            owner = st.selectbox("Владелец", owner_options) if len(owner_options) > 0 else st.selectbox("Владелец", ["First Owner", "Second Owner"])
            seats = st.selectbox("Количество мест", seats_options) if len(seats_options) > 0 else st.selectbox("Количество мест", [5, 7, 8])
            # Выбор марки: из списка или ввод своей
            brand_choice = st.radio("Выберите марку:", ["Из списка", "Ввести свою марку"])
            
            if brand_choice == "Из списка":
                brand = st.selectbox("Марка (из списка)", brand_options)
            else:
                brand = st.text_input("Введите марку автомобиля", "Tesla")
        
        if st.button("Предсказать цену", type="primary"):
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
            
            input_df = pd.DataFrame([input_dict])
            processed_df = preprocess_input(input_df)
            
            for col in data['feature_names']:
                if col not in processed_df.columns:
                    processed_df[col] = 0
            
            processed_df = processed_df[data['feature_names']]
            prediction = model.predict(processed_df)[0]
            
            st.success(f"### Предсказанная цена: {prediction:,.2f} (возможны неточности, обратитесь к специалисту)")
            
    
    with tab2:
        st.header("Пакетное предсказание из CSV файла")
        
        uploaded_file = st.file_uploader("Загрузите CSV файл с данными об автомобилях", type="csv")
        
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            
            required_cols = ['name', 'year', 'km_driven', 'fuel', 'seller_type', 
                            'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
            
            missing_cols = [col for col in required_cols if col not in batch_df.columns]
            
            if missing_cols:
                st.error(f"Отсутствуют колонки: {missing_cols}")
            else:
                st.success(f"Загружено {len(batch_df)} автомобилей")
                
                if st.button("Обработать и предсказать", type="primary"):
                    with st.spinner("Обработка"):
                        original_df = batch_df.copy()
                        processed_batch = preprocess_input(batch_df)
                        
                        for col in data['feature_names']:
                            if col not in processed_batch.columns:
                                processed_batch[col] = 0
                        
                        processed_batch = processed_batch[data['feature_names']]
                        predictions = model.predict(processed_batch)
                        
                        original_df['predicted_price'] = predictions
                        
                        st.subheader("Результаты предсказаний")
                        st.dataframe(original_df[['name', 'year', 'fuel', 'engine', 'predicted_price']], 
                                   use_container_width=True)
                        
                        csv = original_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Скачать CSV с предсказаниями",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
        
        with st.expander("Формат CSV файла"):
            st.write("CSV файл должен содержать следующие колонки:")
            st.code("""
name, year, km_driven, fuel, seller_type, transmission, 
owner, mileage, engine, max_power, seats, torque
            """)
            st.write("Пример:")
            example_df = pd.DataFrame({
                'name': ['Maruti Suzuki Swift', 'Hyundai i20'],
                'year': [2018, 2020],
                'km_driven': [45000, 25000],
                'fuel': ['Petrol', 'Diesel'],
                'seller_type': ['Individual', 'Dealer'],
                'transmission': ['Manual', 'Automatic'],
                'owner': ['First Owner', 'First Owner'],
                'mileage': [20.4, 18.0],
                'engine': [1197, 1493],
                'max_power': [82.0, 100.0],
                'seats': [5, 5],
                'torque': ['113Nm@4400rpm', '240Nm@1500rpm']
            })
            st.dataframe(example_df, use_container_width=True)
    
    with tab3:
        st.header("EDA")
        
        if 'X_train' in data and 'y_train' in data:
            X_train = data['X_train']
            y_train = data['y_train']
            
            st.subheader("Обзор датасета")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Всего образцов", len(X_train))
            with col2:
                st.metric("Количество признаков", len(X_train.columns))
            with col3:
                st.metric("Средняя цена", f"{y_train.mean():,.0f}")
            
            st.subheader("Распределение цен")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(y_train, bins=50, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Цена')
            ax.set_ylabel('Количестов')
            ax.set_title('Распределение цен авто')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            st.subheader("Числовые признаки vs Цена")
            num_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power']
            num_cols = [col for col in num_cols if col in X_train.columns]
            
            selected_num = st.selectbox("Выберите признак", num_cols)
            
            if selected_num:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(X_train[selected_num], y_train, alpha=0.5, s=10)
                ax.set_xlabel(selected_num)
                ax.set_ylabel('Price')
                ax.set_title(f'{selected_num} vs Price')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            st.subheader("Анализ категориальных признаков")
            cat_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'brand', 'seats']
            cat_cols = [col for col in cat_cols if col in X_train.columns]
            
            selected_cat = st.selectbox("Выберите призгнак", cat_cols)
            
            if selected_cat:
                fig, ax = plt.subplots(figsize=(12, 6))
                cat_stats = pd.DataFrame({
                    'category': X_train[selected_cat],
                    'price': y_train
                })
                cat_means = cat_stats.groupby('category')['price'].mean().sort_values(ascending=False)
                
                if len(cat_means) > 20:
                    cat_means = cat_means.head(20)
                
                bars = ax.bar(range(len(cat_means)), cat_means.values)
                ax.set_xticks(range(len(cat_means)))
                ax.set_xticklabels(cat_means.index, rotation=45, ha='right')
                ax.set_xlabel(selected_cat)
                ax.set_ylabel('Average Price')
                ax.set_title(f'Average Price by {selected_cat}')
                ax.grid(True, alpha=0.3, axis='y')
                
                for bar, price in zip(bars, cat_means.values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'₹{price:,.0f}', ha='center', va='bottom', fontsize=8)
                
                st.pyplot(fig)
            
            st.subheader("Correlation Heatmap")
            numeric_data = X_train[num_cols].copy()
            numeric_data['price'] = y_train.values
            
            if len(numeric_data.columns) > 1:
                corr = numeric_data.corr()
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                           center=0, square=True, ax=ax)
                ax.set_title('Матрица корреляций')
                st.pyplot(fig)
            
            st.subheader("Статистика признаков")
            stats_df = X_train.describe().T
            stats_df['count'] = stats_df['count'].astype(int)
            st.dataframe(stats_df, use_container_width=True)
    with tab4:
        st.subheader("Веса модели")
        try:
            ridge_model = model.named_steps['model']
            preprocessor = model.named_steps['preprocessor']
            if hasattr(preprocessor, 'get_feature_names_out'):
                feature_names = preprocessor.get_feature_names_out()
            if hasattr(ridge_model, 'coef_'):
                coef = ridge_model.coef_
                coef_df = pd.DataFrame({
                    'feature': feature_names[:len(coef)],
                    'coefficient': coef
                }).sort_values('coefficient', key=abs, ascending=False)
                top_n = min(20, len(coef_df))
                fig, ax = plt.subplots(figsize=(12, 8))
                
                top_coef = coef_df.head(top_n)
                colors = ['red' if x < 0 else 'green' for x in top_coef['coefficient']]
                
                bars = ax.barh(range(len(top_coef)), top_coef['coefficient'], color=colors)
                ax.set_yticks(range(len(top_coef)))
                ax.set_yticklabels(top_coef['feature'])
                ax.set_xlabel('Вес признака')
                ax.set_title(f'Топ-{top_n} наиболее значимых признаков')
                ax.grid(True, alpha=0.3, axis='x')
        
                for i, (bar, val) in enumerate(zip(bars, top_coef['coefficient'])):
                    ax.text(val, i, f'{val:.4f}', 
                           va='center', ha='left' if val > 0 else 'right',
                           fontsize=9, color='black')
                
                st.pyplot(fig)
                
                with st.expander("Смотрим все веса"):
                    st.dataframe(coef_df, use_container_width=True)
                
        
        except Exception as e:
             st.warning(f"Could not visualize model coefficients: {str(e)}")

except FileNotFoundError:
    st.error("Не найдены необходимые файлы. Загрузите:")
    st.write("- simple_model.pkl")
    st.write("- preprocessed_data.pkl")

except Exception as e:
    st.error(f"Ошибка загрузки данных: {str(e)[:200]}")
