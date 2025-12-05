import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

st.set_page_config(
    page_title="Предсказание цен на автомобили",
    layout="wide"
)

st.title('EDA')
st.write("Cмотрим графики")


#Пришлось заново строить предобработку, потому что streamlit c ней не подружился
def preprocess_data(df, reference_df=None):
    """Вся предобработка, которую мы использовали"""
    df = df.copy()
    columns_to_clean = ['mileage', 'engine', 'max_power']
    for column in columns_to_clean:
        df[column] = df[column].astype(str).str.replace(r'[^\d\.]+', '', regex=True)
        df[column] = pd.to_numeric(df[column], errors='coerce').astype('float64')
    if 'torque' in df.columns:
        df = df.drop('torque', axis=1)
    if 'name' in df.columns:
        df['brand'] = df['name'].str.split().str[0]
        df = df.drop('name', axis=1)
        col = "brand"
        if reference_df is not None:
            train_cats = set(reference_df[col].unique())
            test_cats = set(df[col].unique())
            unknown = test_cats - train_cats
            if unknown:
                most_frequent = reference_df[col].mode()[0]
                df[col] = df[col].replace(list(unknown), most_frequent)
    
    return df


@st.cache_resource
def load_pipeline():
    """Загружает обученный пайплайн"""
    try:
        # Проверяем, существует ли файл локально
        if os.path.exists('final_pipeline.pkl'):
            with open('final_pipeline.pkl', 'rb') as f:
                pipeline = pickle.load(f)
            st.success("Модель загружена успешно!")
            return pipeline
        else:
            # Если файла нет локально, пробуем загрузить из текущей директории
            st.warning("Файл final_pipeline.pkl не найден в текущей директории")
            st.info("Текущая рабочая директория: " + os.getcwd())
            st.info("Содержимое директории: " + str(os.listdir('.')))
            return None
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None
@st.cache_data
def load_and_preprocess_data():
    try:
        # Загружаем исходные данные
        train_url = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv"
        test_url = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv"
        
        df_train_raw = pd.read_csv(train_url)
        df_test_raw = pd.read_csv(test_url)
        
        y_train = df_train_raw['selling_price']
        y_test = df_test_raw['selling_price']
        X_train_raw = df_train_raw.drop('selling_price', axis=1)
        X_test_raw = df_test_raw.drop('selling_price', axis=1)
        
        return X_train_raw, y_train, X_test_raw, y_test
        
    except Exception as e:
        st.error(f"Ошибка загрузки данных: {e}")
        return None, None, None, None

pipeline = load_pipeline()
X_train_raw, y_train, X_test_raw, y_test = load_and_preprocess_data()

# EDA раздел
st.header('Анализ данных')

if pipeline is not None and X_train_raw is not None:
    with st.spinner("Применяем предобработку данных..."):
        try:
            X_train_cleaned = preprocess_data(X_train_raw)
            X_test_cleaned = preprocess_data(X_test_raw, reference_df=X_train_cleaned)
            df_train_processed = X_train_cleaned.copy()
            df_train_processed['selling_price'] = y_train.values
            df_test_processed = X_test_cleaned.copy()
            df_test_processed['selling_price'] = y_test.values
            
            st.success("Предобработка данных выполнена успешно")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Тренировочные данные")
                st.write(f"Количество строк: {df_train_processed.shape[0]}")
                st.write(f"Количество признаков: {df_train_processed.shape[1]}")
                st.write("Типы данных:")
                st.write(df_train_processed.dtypes)
                st.write("Первые 5 строк:")
                st.dataframe(df_train_processed.head())
            
            with col2:
                st.subheader("Тестовые данные")
                st.write(f"Количество строк: {df_test_processed.shape[0]}")
                st.write(f"Количество признаков: {df_test_processed.shape[1]}")
                st.write("Типы данных:")
                st.write(df_test_processed.dtypes)
                st.write("Первые 5 строк:")
                st.dataframe(df_test_processed.head())
            
            # Графики
            st.subheader("Графики по тренировочным данным")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Распределение цен")
                fig, ax = plt.subplots()
                df_train_processed['selling_price'].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
                ax.set_xlabel("Цена (руб)")
                ax.set_ylabel("Количество")
                ax.set_title("Распределение цен (тренировочные данные)")
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                st.write("Сравнение распределений цен")
                fig, ax = plt.subplots()
                
                # Берем логарифм цен для лучшей визуализации
                train_log_price = np.log1p(df_train_processed['selling_price'])
                test_log_price = np.log1p(df_test_processed['selling_price'])
                
                ax.hist(train_log_price, bins=30, alpha=0.7, label='Тренировочные', edgecolor='black')
                ax.hist(test_log_price, bins=30, alpha=0.7, label='Тестовые', edgecolor='black')
                
                ax.set_xlabel("Логарифм цены")
                ax.set_ylabel("Количество")
                ax.set_title("Сравнение распределений цен")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            
            # Статистика по числовым признакам
            st.subheader("Статистика числовых признаков")
            
            # Выбираем числовые признаки
            numeric_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'selling_price']
            available_numeric_cols = [col for col in numeric_cols if col in df_train_processed.columns]
            
            if available_numeric_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Тренировочные данные:")
                    st.write(df_train_processed[available_numeric_cols].describe())
                
                with col2:
                    st.write("Тестовые данные:")
                    st.write(df_test_processed[available_numeric_cols].describe())
            
            # Матрица корреляций
            st.subheader("Матрица корреляций (тренировочные данные)")
            
            # Берем только числовые колонки
            numeric_df = df_train_processed.select_dtypes(include=[np.number])
            
            if len(numeric_df.columns) > 1:
                fig, ax = plt.subplots(figsize=(10, 8))
                corr_matrix = numeric_df.corr()
                
                # Маска для верхнего треугольника
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                
                sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                           center=0, ax=ax, square=True, cbar_kws={"shrink": 0.8})
                ax.set_title("Корреляция между числовыми признаками")
                plt.xticks(rotation=45, ha='right')
                plt.yticks(rotation=0)
                st.pyplot(fig)
            
            # Информация о пропущенных значениях
            st.subheader("Информация о пропущенных значениях")
            
            missing_train = df_train_processed.isnull().sum()
            missing_test = df_test_processed.isnull().sum()
            
            missing_df = pd.DataFrame({
                'Тренировочные_пропуски': missing_train,
                'Тестовые_пропуски': missing_test
            })
            
            st.write("Пропущенные значения после предобработки:")
            if (missing_df.sum(axis=1) > 0).any():
                st.dataframe(missing_df[missing_df.sum(axis=1) > 0])
            else:
                st.write("Пропущенных значений нет")
            
            # Распределение категориальных признаков
            st.subheader("Распределение категориальных признаков")
            
            categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'brand', 'seats']
            available_cat_cols = [col for col in categorical_cols if col in df_train_processed.columns]
            
            for col in available_cat_cols:
                fig, ax = plt.subplots(figsize=(10, 4))
                value_counts = df_train_processed[col].value_counts().head(10)
                value_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black', alpha=0.7)
                ax.set_title(f"Распределение {col} (топ-10)")
                ax.set_xlabel(col)
                ax.set_ylabel("Количество")
                plt.xticks(rotation=45, ha='right')
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Ошибка при предобработке: {e}")
            df_train_processed = None
            df_test_processed = None
else:
    st.warning("Не удалось загрузить данные или пайплайн")

st.header('Предсказание цены')

pred_mode = st.radio("Способ ввода:", ["CSV файл", "Ручной ввод"], horizontal=True)

if pipeline is not None:
    if pred_mode == "CSV файл":
        pred_file = st.file_uploader("Загрузите CSV для предсказания", type=['csv'], key='pred')
        
        if pred_file is not None:
            df_pred = pd.read_csv(pred_file)
            st.write(f"Загружено {len(df_pred)} автомобилей")
            st.write("Первые 5 строк:")
            st.dataframe(df_pred.head())
            
            if st.button("Предсказать все цены", type="primary"):
                with st.spinner("Выполняется предсказание..."):
                    try:
                        predictions = pipeline.predict(df_pred)
                        df_pred['predicted_price'] = predictions
                        
                        st.success(f"Предсказано {len(predictions)} цен")
                        
                        st.subheader("Результаты предсказания")
                        st.dataframe(df_pred[['predicted_price']].style.format({'predicted_price': '{:,.0f}'}))
                        
                        # Скачивание результатов
                        csv = df_pred.to_csv(index=False)
                        st.download_button(
                            "Скачать результаты",
                            csv,
                            "car_predictions.csv",
                            "text/csv",
                            key='download-csv'
                        )
                    except Exception as e:
                        st.error(f"Ошибка: {str(e)}")
    
    else:  # Ручной ввод
        st.subheader("Введите характеристики автомобиля")
        
        left, right = st.columns(2)
        
        with left:
            name = st.text_input("Марка и модель", "Maruti Swift VDI")
            year = st.slider("Год выпуска", 1990, 2024, 2018)
            km_driven = st.number_input("Пробег (км)", 0, 500000, 50000, step=1000)
            fuel = st.selectbox("Топливо", ["Diesel", "Petrol", "CNG", "LPG"])
            seller_type = st.selectbox("Продавец", ["Individual", "Dealer", "Trustmark Dealer"])
        
        with right:
            transmission = st.selectbox("Коробка передач", ["Manual", "Automatic"])
            owner = st.selectbox("Владелец", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"])
            mileage = st.number_input("Расход топлива", 5.0, 30.0, 18.5, step=0.1)
            engine = st.number_input("Объем двигателя (CC)", 600, 5000, 1200, step=100)
            max_power = st.number_input("Мощность (bhp)", 30.0, 500.0, 80.0, step=5.0)
            seats = st.selectbox("Количество мест", [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        
        if st.button("Узнать цену", type="primary"):
            # Создаем DataFrame с введенными данными
            input_data = pd.DataFrame([{
                'name': name,
                'year': year,
                'km_driven': km_driven,
                'fuel': fuel,
                'seller_type': seller_type,
                'transmission': transmission,
                'owner': owner,
                'mileage': mileage,
                'engine': engine,
                'max_power': max_power,
                'seats': seats
            }])
            
            # Показываем введенные данные
            st.write("Введенные данные:")
            st.dataframe(input_data)
            
            try:
                # Делаем предсказание
                prediction = pipeline.predict(input_data)[0]
                
                st.markdown("---")
                st.subheader("Результат предсказания")
                st.markdown(f"### **Предсказанная цена:** {prediction:,.0f} руб.")
                
                # Дополнительная информация
                st.info(f"Это примерно {prediction/100000:.1f} лакхов")
                
            except Exception as e:
                st.error(f"Ошибка предсказания: {str(e)}")
                st.write("Убедитесь, что все поля заполнены корректно.")

# Визуализация весов модели
st.header('Важность признаков модели')

if pipeline is not None and st.checkbox("Показать важность признаков", value=False):
    try:
        if hasattr(pipeline, 'named_steps') and 'model' in pipeline.named_steps:
            model = pipeline.named_steps['model']
            
            if hasattr(model, 'coef_'):
                # Получаем имена признаков из пайплайна
                try:
                    # Пробуем получить имена признаков через preprocessor
                    preprocessor = pipeline.named_steps['preprocessor']
                    # Получаем имена признаков после всех трансформаций
                    feature_names = []
                    
                    # Для ColumnTransformer
                    if hasattr(preprocessor, 'get_feature_names_out'):
                        feature_names = preprocessor.get_feature_names_out()
                    else:
                        # Если не работает, создаем имена вручную
                        # Примерное количество признаков после OHE
                        num_features = len(model.coef_)
                        feature_names = [f'feature_{i}' for i in range(num_features)]
                except:
                    feature_names = [f'feature_{i}' for i in range(len(model.coef_))]
                
                coef = model.co_
                
                # Создаем DataFrame с весами
                weights_df = pd.DataFrame({
                    'Признак': feature_names,
                    'Вес': coef,
                    'Абсолютное_значение': np.abs(coef)
                }).sort_values('Абсолютное_значение', ascending=False)
                
                # Топ-15 признаков для визуализации
                top_weights = weights_df.head(15)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                y_pos = np.arange(len(top_weights))
                colors = ['green' if x > 0 else 'red' for x in top_weights['Вес']]
                
                ax.barh(y_pos, top_weights['Вес'], color=colors, alpha=0.7, edgecolor='black')
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top_weights['Признак'], fontsize=9)
                ax.set_xlabel("Вес признака")
                ax.set_title("Топ-15 самых важных признаков (Ridge регрессия)")
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
                ax.grid(axis='x', alpha=0.3)
                
                st.pyplot(fig)
                
                # Таблица с весами
                st.subheader("Таблица весов признаков")
                st.dataframe(weights_df.head(20))
                
            else:
                st.write("Модель не имеет атрибута coef_")
        else:
            st.write("Не удалось извлечь модель из пайплайна")
    except Exception as e:
        st.error(f"Ошибка при визуализации весов: {str(e)}")

# Боковая панель
with st.sidebar:
    st.header("Информация о модели")
    st.markdown("""
    **Характеристики модели:**
    
    📊 **Алгоритм:** Ridge регрессия
    
    🎯 **Точность (R²):** 0.69
    
    ✅ **31% предсказаний** в пределах 10% от реальной цены
    
    🔧 **Предобработка включает:**
    - Очистка числовых полей
    - Извлечение бренда из названия
    - Заполнение пропусков медианой/модой
    - Стандартизация числовых признаков
    - One-Hot Encoding категориальных
    
    📈 **Данные для EDA:**
    - Тренировочная выборка: 7,000+ автомобилей
    - Тестовая выборка: 2,000+ автомобилей
    """)
    
    st.markdown("---")
    
    if pipeline is not None:
        st.success("✅ Модель загружена успешно")
    else:
        st.error("❌ Модель не загружена")
    
    st.markdown("---")
    st.markdown("**Инструкция:**")
    st.markdown("""
    1. Используйте вкладку EDA для анализа данных
    2. Выберите способ ввода данных для предсказания
    3. Нажмите кнопку для получения предсказания
    4. Изучите важность признаков модели
    """)

# Футер
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Модель машинного обучения для предсказания цен на автомобили</p>
    <p>Для корректной работы убедитесь, что файл <code>final_pipeline.pkl</code> находится в рабочей директории</p>
</div>
""", unsafe_allow_html=True)
